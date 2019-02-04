import os
import numpy as np
#import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras.utils import to_categorical
from keras import backend as K
from keras.models import model_from_json
from Preprocessing.prep import mk_data, my_log

def simple_model(dm=True):
    """
    @params: dm (boolean)
        - Displays model architecture.
    Creates a two-layered simple network:
    -------------------------------------
    Layer 1:      Dense, Fully connected input layer
    Activation 1: Sigmoid
    Layer 2:      Dense, fully connected output layer
    Activation 2: Softmax
    -------------------------------------
    """
    model = Sequential()
    model.add(Dense(32, input_shape=(28, 28, 1)))
    model.add(Activation("sigmoid"))
    model.add(Flatten())
    model.add(Dense(10))
    model.add(Activation("softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy",
                  metrics=["accuracy"])
    if dm:
        model.summary()
    return model

def custom_run(batch_size, epochs, validation_frac=None, model_name="simple_mnist", display_model=True, log=True):
    """
    @params: batch_size (int), epocs (int), model_name (str), display_model (boolean), log (booean)
    validation frac (int)
        - how much of the training set is to be sliced of and used as validation.
        - validation_frac=2 --> 1/2 of the training set
        
    This function trains and saves simple_model()
    """
    
    # Getting data.
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = mk_data(validation=validation_frac, log=True)
    # Getting model.
    model = simple_model(dm=display_model)
    # Trains and stores to var: history
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
          verbose=2, validation_data=(X_val, y_val))
    # Calc scores on test data
    score = model.evaluate(X_test, y_test, verbose=1)
    if log:
        path = "logfile.txt"
        my_log(path, "Metrics:\n------------------------------\n")
        my_log(path, "Test loss: {}\n".format(score[0]))
        my_log(path, "Test acc:  {}\n".format(score[1]))
        my_log(path, "------------------------------\n")
    else:
        print("Test loss: {}\n".format(score[0]))
        print("Test acc:  {}\n".format(score[0]))
    model.save(model_name + ".hd5")
    # save to JSON
    model_json = model.to_json()
    with open(model_name + ".json", "w") as json_file:
        json_file.write(model_json)
    # save weights to HDF5
    model.save_weights(model_name + "_weights.h5")
    if log:
        my_log(path, "save:\n------------------------------\n")
        my_log(path, "Saved and trained model: {}\n".format(model_name))
    return history

def my_plot(history):
    """
    @params: history (data from training, dict)

    Plots training and validation set loss and accuracy
    """
    fig = plt.figure()
    plt.subplot(2,1,1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.subplot(2,1,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.tight_layout()
    return plt.show()

def main():
    # Hyperparameters
    batch_size=64
    epochs=10
    # Training
    metrics = custom_run(batch_size, epochs, validation_frac=12, display_model=True, log=True)
    # Plot.
    my_plot(metrics)
    

if __name__ == "__main__":
    main()
    
