'''
TU/e BME Project Imaging 2021
Convolutional neural network for PCAM
Author: Suzanne Wetstein
'''

# disable overly verbose tensorflow logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}   
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

# unused for now, to be used for ROC analysis
from sklearn.metrics import roc_curve, auc


# assuming this file is in the 'code' directory, which is in the same folder as the 'Data' directory:


#         |---- code-- cnn.py
# parent--|                       |-- train
#         |---- Data-- train+val--|
#         |                       |-- validation
#         |---- logs
#         |
#         |---- metadata

# either open the parent directory directly of use following line to change the os path to parent from this file:
# os.chdir('..')

dir=os.getcwd()

# the size of the images in the PCAM dataset
IMAGE_SIZE = 96


def get_pcam_generators(base_dir, train_batch_size=100, val_batch_size=32):

     # dataset parameters
     train_path = os.path.join(base_dir, 'train+val','train')
     valid_path = os.path.join(base_dir, 'train+val','valid')


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


def get_model_exercise_1(kernel_size=(3,3), pool_size=(4,4), first_filters=32, second_filters=64):

     # build the model
     first_model = Sequential()

     first_model.add(Conv2D(first_filters, kernel_size, activation = 'relu', padding = 'same', input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)))
     first_model.add(MaxPool2D(pool_size = pool_size))

     first_model.add(Conv2D(second_filters, kernel_size, activation = 'relu', padding = 'same'))
     first_model.add(MaxPool2D(pool_size = pool_size))

     first_model.add(Flatten())
     first_model.add(Dense(64, activation = 'relu'))
     first_model.add(Dense(1, activation = 'sigmoid'))


     # compile the model
     first_model.compile(SGD(learning_rate=0.01, momentum=0.95), loss = 'binary_crossentropy', metrics=['accuracy'])

     return first_model

# get the model
model_exercise_1 = get_model_exercise_1()

def get_model_covolutional(kernel_size=(3,3), pool_size=(4,4), first_filters=32, second_filters=64, third_filters=128, fourth_filters=264):

     # build the model
     model_convolutional = Sequential()

     model_convolutional.add(Conv2D(first_filters, kernel_size, activation = 'relu', padding = 'same', input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)))
     model_convolutional.add(MaxPool2D(pool_size = pool_size))

     model_convolutional.add(Conv2D(second_filters, kernel_size, activation = 'relu', padding = 'same'))
     model_convolutional.add(MaxPool2D(pool_size = pool_size))

     model_convolutional.add(Conv2D(third_filters, kernel_size, activation = 'relu', padding = 'same'))
     model_convolutional.add(MaxPool2D(pool_size = pool_size))
     
     # Fully convolutional replacement for Dense layer
     model_convolutional.add(Conv2D(1, (1,1), activation='sigmoid'))  # 1x1 conv acts like a dense layer
     model_convolutional.add(GlobalAveragePooling2D())  # Averages over all spatial locations

     # compile the model
     model_convolutional.compile(SGD(learning_rate=0.01, momentum=0.95), loss = 'binary_crossentropy', metrics=['accuracy'])

     return model_convolutional

# get the model
model_convolutional = get_model_covolutional()


# training of the model. Change this to the type of model you want to train.
model=model_exercise_1 # or model_convolutional


# get the data generators
train_gen, val_gen = get_pcam_generators( dir+'/Data') # change this to the path of your data directory if it does not follow abovementioned structure.



# save the model and weights to the folder 'metadata'
model_name='CNN_model'
model_filepath = 'metadata/'+model_name + '.json'
weights_filepath = 'metadata/'+model_name + '_weights.keras'


model_json = model.to_json() # serialize model to JSON
with open(model_filepath, 'w') as json_file:
    json_file.write(model_json)


# define the model checkpoint and Tensorboard callbacks to directory 'logs'
checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tensorboard = TensorBoard(os.path.join('logs', model_name))
callbacks_list = [checkpoint, tensorboard]


# train the model
train_steps = train_gen.n//train_gen.batch_size
val_steps = val_gen.n//val_gen.batch_size

# get the history of the model you want to use, if you want to get the history of the model only using convolutional layers, change 'model_exercise_1' to 'model_convolutiona'
history = model.fit(train_gen, steps_per_epoch=train_steps,
                    validation_data=val_gen,
                    validation_steps=val_steps,
                    epochs=3,
                    callbacks=callbacks_list)



def plot_roc_curve(y_true, y_pred_prob):
    """
    This function computes the ROC curve and plots it.

    Parameters:
    - y_true: The true labels (0 or 1) of the dataset.
    - y_pred_prob: The predicted probabilities for each image.

    Returns:
    - The AUC (Area Under Curve) score.
    """
    # Compute the ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    
    # Compute the AUC
    roc_auc = auc(fpr, tpr)
    
    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
    # Return the AUC value
    return roc_auc

# ROC analysis
y_pred_prob = model.predict(val_gen)  # Returns a probability per image from the model that you select, if you want to use another model, you should change "model_exercise_1" to the model you want to use
y_true = val_gen.classes  # The real labels (0 or 1) from the validation set
plot_roc_curve(y_true, y_pred_prob)



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

plot_history(history, title = 'CNN model only using convolutional layers') # change the "history" to the correct history you want to plot