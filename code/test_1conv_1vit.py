import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}   
import tensorflow as tf

import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, GlobalAveragePooling1D, DepthwiseConv2D, MaxPooling2D
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
from tensorflow.keras.optimizers import SGD
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

def transformer_block(x, embed_dim, num_heads=8, ff_dim=256, dropout_rate=0.1): 
    # Pre-normalization
    x = LayerNormalization(epsilon=1e-6)(x)

    # Multi-head attention layer
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
    attn_output = Dropout(dropout_rate)(attn_output) # dropout to prevent overfitting

    # Post-normalization
    x = LayerNormalization(epsilon=1e-6)(x + attn_output)  
    
    # Feed-forward network (Multi Layer Perceptron)
    ffn_output = tf.keras.Sequential([
        Dense(ff_dim, activation="gelu"), # expands feature dimension and introduces non-linearity (to recognize complex patterns)
        Dense(embed_dim) # projects back to original size
    ])(x)
    ffn_output = Dropout(dropout_rate)(ffn_output) # dropout for generalization

    output=LayerNormalization(epsilon=1e-6)(x + ffn_output)

    return output

# testing with one MBConv and one ViT layer with differing metalayers
def inverted_residual_block(input, expand=64, squeeze=16):
    m = Conv2D(expand, (1,1), activation=None, padding='same', use_bias=False)(input)

    m = DepthwiseConv2D((3,3), activation=None, padding='same', use_bias=False)(m)

    output = Conv2D(squeeze, (1,1), activation=None, padding='same', use_bias=False)(m)
    output = BatchNormalization()(output)

    # Residual Connection (Skip Connection)
    shortcut = input
    if input.shape[-1] != squeeze: 
        shortcut = Conv2D(squeeze, (1,1), padding='same', use_bias=False)(input)
    
    output = Add()([shortcut, output])  
    output = ReLU()(output)  

    return output

# def CoAtNet(input_shape, 
#             MBConv1_expand=64, MBConv1_squeeze=3,  
#             num_heads1=4,
#             num_classes=1):
    
#     inputs = Input(shape=input_shape)
    
#     # 2x CNN block
#     x = inverted_residual_block(inputs, MBConv1_expand, MBConv1_squeeze) 

#     #x = Conv2D(8, (3,3), strides=2, padding='same', activation='relu')(x)
#     x = MaxPooling2D(pool_size=(4,4))(x)

#     # Automatic reshaping
#     x = Reshape((-1, x.shape[-1]))(x)

#     # 2x transformer block
#     x = transformer_block(x, embed_dim=x.shape[-1], num_heads=num_heads1)
    
#     x = GlobalAveragePooling1D()(x)
#     outputs = Dense(num_classes, activation="sigmoid")(x)
    
#     return Model(inputs, outputs)

def CoAtNet(input_shape, 
            MBConv1_expand=64, MBConv1_squeeze=16, 
            MBConv2_expand=32, MBConv2_squeeze=8, 
            num_heads1=4, num_heads2=4, 
            num_classes=1):
    
    inputs = Input(shape=input_shape)
    
    # 2x CNN block
    x = inverted_residual_block(inputs, MBConv1_expand, MBConv1_squeeze) 
    x = inverted_residual_block(x, MBConv2_expand , MBConv2_squeeze)

    x = Conv2D(32, (3,3), strides=2, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(4,4))(x)
    
    # Automatic reshaping
    x = Reshape((-1, x.shape[-1]))(x)

    # 2x transformer block
    x = transformer_block(x, embed_dim=x.shape[-1], num_heads=num_heads1)
    x = transformer_block(x, embed_dim=x.shape[-1], num_heads=num_heads2)
    
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(num_classes, activation="sigmoid")(x)
    
    return Model(inputs, outputs)

input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
model = CoAtNet(input_shape)
model.compile(SGD(learning_rate=0.001, momentum=0.95), loss = 'binary_crossentropy', metrics=['accuracy'])
model_name = 'CoAtNet_2MBConv_4x4pool_2ViT_10epochs'


# save the model and weights
model_filepath = dir+'/metadata/'+model_name + '.json'
weights_filepath = dir+'/metadata/'+model_name + '_weights.keras'

model_json = model.to_json() # serialize model to JSON
with open(model_filepath, 'w') as json_file:
    json_file.write(model_json)


# get the data generators
train_gen, val_gen = get_pcam_generators(dir+'\Data')


# define the model checkpoint and Tensorboard callbacks
checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tensorboard = TensorBoard(os.path.join('logs', model_name))
callbacks_list = [checkpoint, tensorboard]


# since the model is trained for only 10 "mini-epochs", i.e. half of the data is
# not used during training
train_steps = train_gen.n//train_gen.batch_size
val_steps = val_gen.n//val_gen.batch_size

#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
history = model.fit(train_gen, steps_per_epoch=train_steps,
                    validation_data=val_gen,
                    validation_steps=val_steps,
                    epochs=50,
                    callbacks=callbacks_list)

from sklearn.metrics import accuracy_score, recall_score
y_pred_prob = model.predict(val_gen)  # Returns a probability per image
# Zet de voorspelde waarschijnlijkheden om naar binaire labels (0 of 1)
y_pred = (y_pred_prob > 0.5).astype(int)  # Binaire voorspelling op basis van 0.5 drempel

y_true = val_gen.classes  # The real labels (0 or 1) from the validation set

accuracy = accuracy_score(y_true, y_pred)
recall = recall_score(y_true, y_pred, average='binary') 

print(f'Accuracy: {accuracy:.4f}')
print(f'Recall: {recall:.4f}')