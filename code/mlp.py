"""
TU/e BME Project Imaging 2021
Simple multiLayer perceptron code for MNIST
Author: Suzanne Wetstein
"""

# disable overly verbose tensorflow logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf


# import required packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard

"""
def data_preparation():
    # load the dataset using the builtin Keras method
    (X_train, y_train), (X_test, y_test) = mnist.load_data()


    # derive a validation set from the training set
    # the original training set is split into 
    # new training set (90%) and a validation set (10%)
    X_train, X_val = train_test_split(X_train, test_size=0.10, random_state=101)
    y_train, y_val = train_test_split(y_train, test_size=0.10, random_state=101)



    # the shape of the data matrix is NxHxW, where
    # N is the number of images,
    # H and W are the height and width of the images
    # keras expect the data to have shape NxHxWxC, where
    # C is the channel dimension
    X_train = np.reshape(X_train, (-1,28,28,1)) 
    X_val = np.reshape(X_val, (-1,28,28,1))
    X_test = np.reshape(X_test, (-1,28,28,1))


    # convert the datatype to float32
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    X_test = X_test.astype('float32')


    # normalize our data values to the range [0,1]
    X_train /= 255
    X_val /= 255
    X_test /= 255


    # convert 1D class arrays to 10D class matrices
    y_train = to_categorical(y_train, 10)
    y_val = to_categorical(y_val, 10)
    y_test = to_categorical(y_test, 10)

    return X_train, X_val, X_test, y_train, y_val, y_test

def model_training(neuron_architecture=[64, 10], hidden_layer_activation='relu',model_name='64_10'):
    

    Train a model

    Args:
    neuron_architecture (list): list with the number of neurons per layer. The last entry is considered as the output layer.
    

    model = Sequential()
    # flatten the 28x28x1 pixel input images to a row of pixels (a 1D-array)
    model.add(Flatten(input_shape=(28,28,1))) 
    # fully connected layers as specified by neuron_architecture
    for layer in neuron_architecture[:-1]:
        model.add(Dense(layer, activation=hidden_layer_activation))
    # output layer with 10 nodes (one for each class) and softmax nonlinearity
    model.add(Dense(neuron_architecture[-1], activation='softmax')) 


    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    # use this variable to name your model
    model_name=model_name

    # create a way to monitor our model in Tensorboard
    tensorboard = TensorBoard("logs/" + model_name)

    # train the model
    model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1, validation_data=(X_val, y_val), callbacks=[tensorboard])

    return model, tensorboard

def model_testing(model):
    score = model.evaluate(X_test, y_test, verbose=0)

    print("Loss: ",score[0])
    print("Accuracy: ",score[1])

    return score



X_train, X_val, X_test, y_train, y_val, y_test = data_preparation()

#================================= EXERCISE 1 =====================================

scores={}

# 1 hidden layer with 64 neurons and relu, 10 output neurons.
model1, tensorboard1=model_training(neuron_architecture=[64, 10], hidden_layer_activation='relu',model_name='64_relu_10')
score1=model_testing(model1)
scores['model1']=score1

# 2 hidden layers with 64 and 32 neurons and relu, 10 output neurons.
model2, tensorboard2=model_training(neuron_architecture=[64, 32, 10], hidden_layer_activation='relu',model_name='64_32_relu_10')
score2=model_testing(model2)
scores['model2']=score2

# 3 hidden layers with 64, 32 and 16 neurons and relu, 10 output neurons.
model3, tensorboard3=model_training(neuron_architecture=[64, 32, 16, 10], hidden_layer_activation='relu',model_name='64_32_16_relu_10')
score3=model_testing(model3)
scores['model3']=score3

# 3 hidden layers with 64, 64 and 64 neurons and relu, 10 output neurons.
model4, tensorboard4=model_training(neuron_architecture=[64, 64, 64, 10], hidden_layer_activation='relu',model_name='64_64_64_relu_10')
score4=model_testing(model4)
scores['model4']=score4

# 1 hidden layer with 10 neurons and relu, 10 output neurons.
model5, tensorboard5=model_training(neuron_architecture=[10, 10], hidden_layer_activation='relu',model_name='10_relu_10')
score5=model_testing(model5)
scores['model5']=score5

# 4 hidden layers with 128, 64, 32 and 16 neurons and relu, 10 output neurons.
model6, tensorboard6=model_training(neuron_architecture=[128,64,32,16,10], hidden_layer_activation='relu',model_name='128_64_32_16_relu_10')
score6=model_testing(model6)
scores['model6']=score6


print(scores)

#================================= EXERCISE 2 =====================================

#no hidden layers, 10 output neurons.
model7, tensorboard7=model_training(neuron_architecture=[10], hidden_layer_activation='relu',model_name='no_hidden_10')
score7=model_testing(model7)
scores['model7']=score7

# 3 hidden layers with 64, 32 and 16 neurons and linear activation, 10 output neurons.
model9, tensorboard9=model_training(neuron_architecture=[64, 32, 16, 10], hidden_layer_activation='linear',model_name='64_32_16_linear_10')
score9=model_testing(model9)
scores['model9']=score9

"""

#================================= EXERCISE 3 =====================================
# load the dataset using the builtin Keras method
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# mapping to 4 labels
mapping = {
        1: 0, 7: 0,  # "vertical digits" - 1, 7 receive label '0'
        0: 1, 6: 1, 8: 1, 9: 1,  # "loopy digits" - 0, 6, 8, 9 receive label '1'
        2: 2, 5: 2,  # "curly digits" - 2, 5 receive label '2'
        3: 3, 4: 3   # "other" - 3, 4 receive label '3'
    }

y_train = np.array([mapping[label] for label in y_train])
y_test = np.array([mapping[label] for label in y_test])


# derive a validation set from the training set
# the original training set is split into 
# new training set (90%) and a validation set (10%)
X_train, X_val = train_test_split(X_train, test_size=0.10, random_state=101)
y_train, y_val = train_test_split(y_train, test_size=0.10, random_state=101)


# the shape of the data matrix is NxHxW, where
# N is the number of images,
# H and W are the height and width of the images
# keras expect the data to have shape NxHxWxC, where
# C is the channel dimension
X_train = np.reshape(X_train, (-1,28,28,1)) 
X_val = np.reshape(X_val, (-1,28,28,1))
X_test = np.reshape(X_test, (-1,28,28,1))


# convert the datatype to float32
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')


# normalize our data values to the range [0,1]
X_train /= 255
X_val /= 255
X_test /= 255


# convert 1D class arrays to 4D class matrices, since we now only have 4 classes
y_train = to_categorical(y_train, 4)
y_val = to_categorical(y_val, 4)
y_test = to_categorical(y_test, 4)


model = Sequential()
# flatten the 28x28x1 pixel input images to a row of pixels (a 1D-array)
model.add(Flatten(input_shape=(28,28,1))) 
# fully connected layer with 64 neurons and ReLU nonlinearity
model.add(Dense(64, activation='relu'))
# output layer with 32 nodes (one for each class) and softmax nonlinearity
model.add(Dense(32, activation='relu'))
# output layer with 16 nodes (one for each class) and softmax nonlinearity
model.add(Dense(16, activation='relu'))
# output layer with 4 nodes (one for each class) and softmax nonlinearity
model.add(Dense(4, activation='softmax')) 


# compile the model
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# use this variable to name your model
model_name="Classification"

# create a way to monitor our model in Tensorboard
tensorboard = TensorBoard("logs/" + model_name)

# train the model
model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1, validation_data=(X_val, y_val), callbacks=[tensorboard])


score = model.evaluate(X_test, y_test, verbose=0)


print("Loss: ",score[0])
print("Accuracy: ",score[1])
