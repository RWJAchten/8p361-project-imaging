
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}   

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

IMAGE_SIZE=96

def get_pcam_generators(base_dir, train_batch_size=32, val_batch_size=32):
     """
     Gets the data generators
     """

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

def model_preparation(model,model_name,dir,learning_rate=0.0005):
    """
    Activate the model by assigning an optimizer and loss metrics and initialize the callbacks. 
    """

    model.compile(Adam(learning_rate), loss = 'binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Recall()])
 
    # save the model and weights
    model_filepath = dir+'/metadata/'+model_name + '.json'
    weights_filepath = dir+'/metadata/'+model_name + '_weights.hdf5' #'_weights.keras'

    model_json = model.to_json() # serialize model to JSON
    with open(model_filepath, 'w') as json_file:
        json_file.write(model_json)
    
    # define the model checkpoint and Tensorboard callbacks
    checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    tensorboard = TensorBoard(os.path.join('logs', model_name))
    callbacks_list = [checkpoint, tensorboard]
    
    return model, callbacks_list, weights_filepath

def train_model(model, train_gen,val_gen,epochs,callbacks_list):
    """
    Train the model. 
    """
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
    """
    Evaluate the model on the validation set.
    """
    steps = val_gen.samples // val_gen.batch_size

    loss, accuracy, area_under_curve, recall = model.evaluate(val_gen, steps=steps)

    print(f'loss: {loss:.4f}')
    print(f'accuracy: {accuracy:.4f}')
    print(f'AUC: {area_under_curve:.4f}')
    print(f'Recall: {recall:.4f}')

    
    return accuracy, recall, area_under_curve, recall

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

def evaluate_model_from_file(model_file_path, val_gen):
    
    model=tf.keras.models.load_model(model_file_path)
    steps = val_gen.samples // val_gen.batch_size  
    loss, accuracy, area_under_curve, recall = model.evaluate(val_gen, steps=steps)

    print(f'loss: {loss:.4f}')
    print(f'accuracy: {accuracy:.4f}')
    print(f'AUC: {area_under_curve:.4f}')
    print(f'Recall: {recall:.4f}')

    # Initialize counters
    num_batches = 500
    predictions = []
    true = []

    # Iterate through the first 20 batches of the data generator
    for i, (batch_data, batch_labels) in enumerate(val_gen):
        if i >= num_batches:
            break  # Stop after 20 batches
        print('working on batch', i, '/', num_batches)
        # Perform prediction on the batch
        batch_predictions = model.predict(batch_data)  # You can also use model(batch_data, training=False)
    
        # Store predictions for later analysis (optional)
        predictions.append(batch_predictions)
        true.append(batch_labels)

    # After the loop, you can convert the predictions list into a single array (optional)
    predictions = np.concatenate(predictions, axis=0)
    true = np.concatenate(true, axis=0)

    plot_roc_curve(true, predictions)

    return loss, accuracy, area_under_curve, recall

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

    # Recall


    # AUC



    plt.show()