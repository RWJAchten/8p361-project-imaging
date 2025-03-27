
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}   

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
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
    
    return model,callbacks_list

def train_model(model, train_gen,val_gen,epochs,callbacks_list):
    """
    Train the model. 
    """
    # since the model is trained for only 10 "mini-epochs", i.e. half of the data is
    # not used during training
    train_steps = train_gen.n//train_gen.batch_size
    val_steps = val_gen.n//val_gen.batch_size

     # Voeg de ROC Callback toe aan de lijst van callbacks
    callbacks_list.append(ROCCallback(val_gen))

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


class ROCCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_gen):
        super().__init__()
        self.val_gen = val_gen

    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch+1} ended.")  # Toevoegen om te controleren of de callback wordt uitgevoerd

        # Haal de echte labels op
        y_true = self.val_gen.classes  # Ware labels
        
        # Haal de voorspellingen op (waarschijnlijkheden)
        # Zorg ervoor dat je waarschijnlijkheden hebt en niet de uiteindelijke labels
        y_pred = self.model.predict(self.val_gen, verbose=0)  # Modelvoorspellingen (waarschijnlijkheden)
        
        # Als je een binaire classificatie hebt, moet je de voorspellingen voor de positieve klasse nemen
        if y_pred.shape[1] > 1:  # Als je model meer dan 1 output heeft (bijv. softmax)
            y_pred = y_pred[:, 1]  # Neem de voorspellingen voor de positieve klasse
        else:  # Als je model een enkele output heeft (bijv. sigmoid)
            y_pred = y_pred.flatten()  # Zorg ervoor dat y_pred een 1D-array is

        # Bereken de ROC-curve
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)

        # Plot de ROC-curve
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - Epoch {epoch+1}')
        plt.legend(loc="lower right")
        plt.show()

        print(f'Epoch {epoch+1}: AUC = {roc_auc:.4f}')


def plot_history(history, title=''):
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 4, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 4, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    #recall
    recall_key = [key for key in history.history.keys() if 'recall' in key][-1]
    plt.subplot(1, 4, 3)
    plt.plot(history.history[recall_key], label='Train Recall')
    plt.plot(history.history[f'val_{recall_key}'], label='Val Recall')
    plt.title(f'{title} - Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.legend()

    # AUC: Haal de laatste AUC-metrieknaam dynamisch op
    auc_key = [key for key in history.history.keys() if 'auc' in key][-1]
    plt.subplot(1, 4, 4)
    plt.plot(history.history[auc_key], label='Train AUC')
    plt.plot(history.history[f'val_{auc_key}'], label='Val AUC')
    plt.title(f'{title} - AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend()

    plt.show()