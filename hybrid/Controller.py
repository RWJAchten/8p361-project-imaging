
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}   

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve, auc

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
