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

#os.chdir('..')
dir=os.getcwd()


def transformer_block(x, channel, num_heads=8, dropout_rate=0.1): 

    ff_dim=4*channel

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


def CoAtNet(input_shape, 
             channels=[8,16,32],
            dropout_rate=.1,
            num_heads=4,
            num_classes=1):
    
    inputs = Input(shape=input_shape)

    # first convolution to transform 96x96x3 image into 48x48x64
    x = Conv2D(filters=channels[0],kernel_size=(3,3),strides=(2,2),padding='same')(inputs)

    # reshape into 1d image
    batch, height, width, channel = x.shape
    x = Reshape((height*width, channel))(x)

    # 2x transformer block
    x = transformer_block(x, channel=channels[1], num_heads=num_heads, dropout_rate=dropout_rate)
    x = transformer_block(x, channel=channels[2], num_heads=num_heads, dropout_rate=dropout_rate)
   
    #x = GlobalAveragePooling1D(x.shape[1]//32)(x)
    x = Dense(num_classes, activation="sigmoid")(x)
    outputs = Dropout(.1)(x) # dropout to prevent overfitting
     
    return Model(inputs, outputs)


input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
model = CoAtNet(input_shape)
model.compile(Adam(learning_rate=0.1), loss = 'binary_crossentropy', metrics=['accuracy'])
model_name = 'CoAtNet0'


# save the model and weights
model_filepath = dir+'/metadata/'+model_name + '.json'
weights_filepath = dir+'/metadata/'+model_name + '_weights.keras'

model_json = model.to_json() # serialize model to JSON
with open(model_filepath, 'w') as json_file:
    json_file.write(model_json)


# get the data generators
train_gen, val_gen = get_pcam_generators('/Users/nikavredenbregt/Library/CloudStorage/OneDrive-TUEindhoven/TUe/Year 3/Project AI for MIA')


# define the model checkpoint and Tensorboard callbacks
checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tensorboard = TensorBoard(os.path.join('logs', model_name))
callbacks_list = [checkpoint, tensorboard]


# since the model is trained for only 10 "mini-epochs", i.e. half of the data is
# not used during training
train_steps = train_gen.n//train_gen.batch_size
val_steps = val_gen.n//val_gen.batch_size

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
history = model.fit(train_gen, steps_per_epoch=train_steps,
                    validation_data=val_gen,
                    validation_steps=val_steps,
                    epochs=10,
                    callbacks=callbacks_list)

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
print(roc_auc)

from sklearn.metrics import confusion_matrix as cm
cm(y_pred,y_true)