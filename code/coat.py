import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}   
os.chdir('D:\TUE\AI_in_MIA\8p361-project-imaging')
import tensorflow as tf

import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling1D
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.layers import Reshape
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard


IMAGE_SIZE = 96


def get_pcam_generators(base_dir, train_batch_size=32, val_batch_size=32):

     # dataset parameters
     train_path = os.path.join(base_dir,'train')
     valid_path = os.path.join(base_dir,'valid')


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


def convolutional_block(input, kernel_size=(3,3), first_filters=32):
      x = Conv2D(first_filters, kernel_size, padding = 'same')(input)
      x = BatchNormalization()(x)
      return x


def transformer_block(x, embed_dim, num_heads=8, ff_dim=256, dropout_rate=0.1):
    x = Dense(embed_dim)(x)
    
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
    attn_output = Dropout(dropout_rate)(attn_output)
    x = LayerNormalization(epsilon=1e-6)(x + attn_output)  
    
    ffn_output = tf.keras.Sequential([
        Dense(ff_dim, activation="gelu"),
        Dense(embed_dim)
    ])(x)
    
    ffn_output = Dropout(dropout_rate)(ffn_output)
    return LayerNormalization(epsilon=1e-6)(x + ffn_output)


def CoAtNet(input_shape, num_classes=2):
    inputs = Input(shape=input_shape)
    
    # 2x CNN block
    x = convolutional_block(inputs, 32) 
    x = convolutional_block(x, 64)
    
    x = Reshape((IMAGE_SIZE * IMAGE_SIZE, 32))(x)  
    # 2x transformer block
    x = transformer_block(x, embed_dim=64, num_heads=4)
    x = transformer_block(x, embed_dim=64, num_heads=4)
    
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(1, activation="sigmoid")(x)

    return Model(inputs, outputs)


# get the model
input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
model = CoAtNet(input_shape)
model.compile(SGD(learning_rate=0.001, momentum=0.95), loss = 'binary_crossentropy', metrics=['accuracy'])
model_name = 'CoAtNet'


# save the model and weights
model_filepath = 'metadata/'+model_name + '.json'
weights_filepath = 'metadata/'+model_name + '_weights.keras'

model_json = model.to_json() # serialize model to JSON
with open(model_filepath, 'w') as json_file:
    json_file.write(model_json)


# get the data generators
train_gen, val_gen = get_pcam_generators('Data')


# define the model checkpoint and Tensorboard callbacks
checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tensorboard = TensorBoard(os.path.join('logs', model_name))
callbacks_list = [checkpoint, tensorboard]


# since the model is trained for only 10 "mini-epochs", i.e. half of the data is
# not used during training
train_steps = train_gen.n//train_gen.batch_size
val_steps = val_gen.n//val_gen.batch_size
history = model.fit(train_gen, steps_per_epoch=train_steps,
                    validation_data=val_gen,
                    validation_steps=val_steps,
                    epochs=3,
                    callbacks=callbacks_list)