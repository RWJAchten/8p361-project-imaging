import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}   

import tensorflow as tf

import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, GlobalAveragePooling1D, DepthwiseConv2D, MaxPooling2D, Conv1D
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Add
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard


IMAGE_SIZE = 96

# =============================================== obtain data generators ===================================

def get_pcam_generators(base_dir, train_batch_size=32, val_batch_size=32):

     # dataset parameters
     train_path = os.path.join(base_dir,'train+val','train')
     valid_path = os.path.join(base_dir,'train+val','valid')


     RESCALING_FACTOR = 1./255

     # instantiate data generators
     datagen = ImageDataGenerator(rescale=RESCALING_FACTOR)

     train_gen = datagen.flow_from_directory(train_path,
                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             batch_size=train_batch_size,
                                             class_mode='binary')

     val_gen = datagen.flow_from_directory(valid_path,
                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             batch_size=val_batch_size,
                                             class_mode='binary')

     return train_gen, val_gen


# ===================================== MBConv and ViT blocks ======================================

def transformer_block(x, channel, num_heads=4, dropout_rate=0.1): 

    ff_dim=2*channel

    # Pre-normalization
    x = LayerNormalization(epsilon=1e-6)(x)
    # Multi-head attention layer
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=channel)(x, x)
    attn_output = Dropout(dropout_rate)(attn_output) # dropout to prevent overfitting

    # Post-normalization
    x = LayerNormalization(epsilon=1e-6)(x + attn_output)  
 
    # Feed-forward network (Multi Layer Perceptron)
    x = Dense(ff_dim, activation="gelu")(x), # expands feature dimension and introduces non-linearity (to recognize complex patterns)
    x = Dropout(dropout_rate)(x) # dropout for generalization
  
    x = Dense(channel)(x) # projects back to original size
    ffn_output = Dropout(dropout_rate)(x) # dropout for generalization
  
    output=LayerNormalization(epsilon=1e-6)(x + ffn_output)
    output=tf.squeeze(output,axis=0)

    return output

# testing with one MBConv and one ViT layer with differing metalayers
def inverted_residual_block(input, indim, channel, expand=4):

    # expand with expand*amount of channels and reduce dimensions by factor 2
    m = Conv2D(filters=expand*indim[-1], kernel_size=(3,3), strides=(2,2), activation=None, padding='same')(input)

    # perform depthwise convolution
    m = DepthwiseConv2D((3,3), activation=None, padding='same', use_bias=False)(m)

    #squeeze to desired amount of channels
    output = Conv2D(channel, (1,1), activation=None, padding='same', use_bias=False)(m)
    output = BatchNormalization()(output)

    output = tf.nn.gelu(output)  

    return output

#======================================== model definitions ===============================================

def CoAtNet(input_shape, 
            channels=[64,96,192,384,768],
            dropout_rate=.1,
            num_heads=8,
            num_classes=1):
    
    inputs = Input(shape=input_shape)

    # first convolution to transform 96x96x3 image into 48x48x64
    x = Conv2D(filters=64,kernel_size=(3,3),strides=(2,2),padding='same')(inputs)

    # 2x CNN block
    x = inverted_residual_block(x, x.shape, channel=channels[1]) 
    x = inverted_residual_block(x, x.shape, channel=channels[2])


    # reduce dimensions by factor two and get right amount of channels
    x = Conv2D(filters=channels[3], kernel_size=(3,3), strides=(2,2), padding='same')(x)
    # reshape into 1d image
    batch, height, width, channel = x.shape
    x = Reshape((height*width, channel))(x)

    # 2x transformer block
    x = transformer_block(x, channel=channels[3], num_heads=num_heads, dropout_rate=dropout_rate)
    x = Conv1D(filters=channels[4], kernel_size=3, strides=2, padding='same')(x)
    x = transformer_block(x, channel=channels[4], num_heads=num_heads, dropout_rate=dropout_rate)
    
    
    #x = GlobalAveragePooling1D(x.shape[1]//32)(x)
    x = Dense(num_classes, activation="sigmoid")(x)
    outputs = Dropout(.1)(x) # dropout to prevent overfitting
    
    
    return Model(inputs, outputs)

def CoAtNet_half(input_shape, 
             channels=[8,16,32],
            dropout_rate=.1,
            num_heads=4,
            num_classes=1):
    
    inputs = Input(shape=input_shape)

    # first convolution to transform 96x96x3 image into 48x48x64
    x = Conv2D(filters=channels[0],kernel_size=(3,3),strides=(2,2),padding='same')(inputs)

    # CNN block
    x = inverted_residual_block(x, x.shape, channel=channels[1]) 


    # reduce dimensions by factor two and get right amount of channels
    x = Conv2D(filters=channels[2], kernel_size=(3,3), strides=(2,2), padding='same')(x)

    # reshape into 1d image
    batch, height, width, channel = x.shape
    x = Reshape((height*width, channel))(x)

    # transformer block
    x = transformer_block(x, channel=channels[2], num_heads=num_heads, dropout_rate=dropout_rate)

    
    # output layer
    x = Dense(num_classes, activation="sigmoid")(x)
    outputs = Dropout(.1)(x) # dropout to prevent overfitting
    
    
    return Model(inputs, outputs)

def simple_vit(input_shape, 
            dropout_rate=.1,
            num_heads=4,
            num_classes=1):
    
    inputs = Input(shape=input_shape)
    x=inputs
    # first transform images to (24x24x8) to save some memory
    x = Conv2D(filters=8,kernel_size=(3,3),strides=(4,4),padding='same')(inputs)

    # flatten the images to 2D
    batch, height, width, channel = x.shape
    x = Reshape((height*width, channel))(x)

    # vision transformer 
    x = transformer_block(x, channel=8, num_heads=num_heads, dropout_rate=dropout_rate)

    # dense output layer and dropout
    x = Dense(num_classes, activation="sigmoid")(x)
    outputs = Dropout(.1)(x) # dropout to prevent overfitting

    return Model(inputs, outputs)

# ============================================== preparing and running the models ================================

def model_preparation(model,model_name,dir):

    model.compile(Adam(learning_rate=0.1), loss = 'binary_crossentropy', metrics=['accuracy'])
 
    # save the model and weights
    model_filepath = dir+'/metadata/'+model_name + '.json'
    weights_filepath = dir+'/metadata/'+model_name + '_weights.keras'

    model_json = model.to_json() # serialize model to JSON
    with open(model_filepath, 'w') as json_file:
        json_file.write(model_json)
    
    # define the model checkpoint and Tensorboard callbacks
    checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    tensorboard = TensorBoard(os.path.join('logs', model_name))
    callbacks_list = [checkpoint, tensorboard]
    
    return model,callbacks_list

def train_model(model, train_gen,val_gen,epochs,callbacks_list):
    # since the model is trained for only 10 "mini-epochs", i.e. half of the data is
    # not used during training
    train_steps = train_gen.n//train_gen.batch_size
    val_steps = val_gen.n//val_gen.batch_size

    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    history = model.fit(train_gen, steps_per_epoch=train_steps,
                    validation_data=val_gen,
                    validation_steps=val_steps,
                    epochs=epochs,
                    callbacks=callbacks_list)

    return model, history

def evaluate_model(model, val_gen):

    from sklearn.metrics import accuracy_score, recall_score
    y_pred_prob = list(model.predict(val_gen).flatten())  # Returns a probability per image
    y_pred=[1 if num >= .5 else 0 for num in y_pred_prob]
    y_true = list(val_gen.classes)  # The real labels (0 or 1) from the validation set

    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='binary') 

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Recall: {recall:.4f}')

    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    
    # Compute the AUC
    roc_auc = auc(fpr, tpr)
    print(f'area under curve: {roc_auc:.4f}')

    from sklearn.metrics import confusion_matrix as cm
    confusion_matrix=cm(y_pred,y_true)
    print(confusion_matrix)

    return accuracy, recall, roc_auc, confusion_matrix

# this code is only executed when this python file is runned directly
if __name__=="__main__":
    # make sure OS is searching the right directory. 
    # os.chdir('..')
    dir=os.getcwd()

    train_gen, val_gen = get_pcam_generators(dir+'\Data')

    input_shape=(96,96,3)
    model = simple_vit(input_shape=input_shape)

    model_name='simple_vit1'
    model, callbacks_list=model_preparation(model,model_name)

    epochs=5
    model, history = train_model(model, train_gen, val_gen, epochs, callbacks_list)