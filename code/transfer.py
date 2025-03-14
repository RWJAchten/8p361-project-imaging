'''
TU/e BME Project Imaging 2021
Convolutional neural network for PCAM
Author: Mitko Veta
'''

# disable overly verbose tensorflow logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}   
import tensorflow as tf


import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input


def get_pcam_generators(base_dir, train_batch_size=32, val_batch_size=32):

     # dataset parameters
     train_path = os.path.join(base_dir, 'train+val', 'train')
     valid_path = os.path.join(base_dir, 'train+val', 'valid')
	 
     # instantiate data generators
     datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

     train_gen = datagen.flow_from_directory(train_path,
                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             batch_size=train_batch_size,
                                             class_mode='binary')

     val_gen = datagen.flow_from_directory(valid_path,
                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             batch_size=val_batch_size,
                                             class_mode='binary')

     return train_gen, val_gen

# the size of the images in the PCAM dataset
IMAGE_SIZE = 96

input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)


input = Input(input_shape)

def train_pretrained_model_for_exercise2(weights='imagenet',model_name='transfer_model_without_dropout',Dropout_condition = False):
    # get the pretrained model, cut out the top layer
    pretrained = MobileNetV2(input_shape=input_shape, include_top=False, weights=weights)

    output = pretrained(input)
    output = GlobalAveragePooling2D()(output)
    if Dropout_condition == False:
        output = Dropout(0.5)(output)
        output = Dense(1, activation='sigmoid')(output)
    else:
        output = Dense(1, activation='sigmoid')(output)

    model = Model(input, output)

    # note the lower lr compared to the cnn example
    model.compile(SGD(learning_rate=0.001, momentum=0.95), loss = 'binary_crossentropy', metrics=['accuracy'])

    # print a summary of the model on screen
    model.summary()

    # get the data generators
    train_gen, val_gen = get_pcam_generators('Data') # change this path to the path of your data directory


    # save the model and weights
    model_filepath = 'metadata/'+model_name + '.json'
    weights_filepath = 'metadata/'+model_name + '_weights.keras'

    model_json = model.to_json() # serialize model to JSON
    with open(model_filepath, 'w') as json_file:
        json_file.write(model_json)


    # define the model checkpoint and Tensorboard callbacks
    checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    tensorboard = TensorBoard(os.path.join('logs', model_name))
    callbacks_list = [checkpoint, tensorboard]


    # train the model, note that we define "mini-epochs"
    train_steps = train_gen.n//train_gen.batch_size//20
    val_steps = val_gen.n//val_gen.batch_size//20

    # since the model is trained for only 10 "mini-epochs", i.e. half of the data is
    # not used during training
    history = model.fit(train_gen, steps_per_epoch=train_steps,
                        validation_data=val_gen,
                        validation_steps=val_steps,
                        epochs=10,
                        callbacks=callbacks_list)
    
    return model, val_gen, history

#=============== Exercise 2 ==================

# train the transfer model once with initial weights from imagenet and once without initial weights

model_imagenet, val_gen_imagenet, history_imagenet=train_pretrained_model_for_exercise2(weights='imagenet',model_name='transfer_model_without_dropout')
model_none, val_gen_none, history_none=train_pretrained_model_for_exercise2(weights=None,model_name='transfer_model_without_dropout_without_initial_weights')
model_dropout, val_gen_none, history_dropout=train_pretrained_model_for_exercise2(weights='imagenet',model_name='transfer_model_with_dropout_with_intial_weights', Dropout_condition = True)

def plot_history(history, title='Training History'):
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# Plot each model's history
plot_history(history_imagenet, title='Without Dropout (Imagenet Weights)')
plot_history(history_none, title='Without Dropout (No Initial Weights)')
plot_history(history_dropout, title='With Dropout (Imagenet Weights)')

