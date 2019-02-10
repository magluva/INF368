import numpy as np
import pickle
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import AveragePooling2D
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.datasets import mnist
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import model_from_json


def lenet_model(u1, u2, dim, n_classes, dm=True):
    """
    :param u1: nodes in first fully connected layer (int)
    :param u2: nodes in second fully connected layer (int)
    :param dim: input shape of first layer (tuple)
    :param n_classes: number of classes (int)
    :param dm: show model architecture (boolean)
    :return: keras.Sequential() compiled model.
    -------------------------------------
    Layer 1:      Conv2D layer
    Activation 1: ReLu
    Layer 2:      Average pooling layer
    Layer 3:      Conv2D layer
    Activation 2: ReLu
    Layer 4:      Average pooling layer
    Layer 5:      Dense, fully connected layer
    Activation 3: ReLu
    Layer 2:      Dense, fully connected output layer
    Activation 4: Softmax
    -------------------------------------
    """
    model = Sequential()
    model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=dim))
    model.add(AveragePooling2D())
    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(AveragePooling2D())
    model.add(Flatten())
    model.add(Dense(units=u1, activation='relu'))
    model.add(Dense(units=u2, activation='relu'))
    model.add(Dense(units=n_classes, activation='softmax'))
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=SGD())
    if dm:
        model.summary()
    return model


def my_plot(history, file_name):
    """
    :param history: training history (dict)
    :param file_name: desired PNG save file name (str)
    :return: matplotlib.pyplot.show()
    -------------------------------------
    Plots train/validation accuracy and
    loss of model.
    -------------------------------------
    """
    fig = plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.tight_layout()
    plt.savefig(file_name + ".png")
    return plt.show()


def save_model(model, model_name, history=None):
    """
    :param model: Sequential() model to save
    :param model_name: desired file name (str)
    :param history: training history (dict)
    :return: None
    -------------------------------------
    Saves model as .json file,
    weights as HDF5 and training history
    as a dict in pickle file format.
    -------------------------------------
    """
    # Save model as json.
    model_json = model.to_json()
    with open(model_name + ".json", "w") as json_file:
        json_file.write(model_json)
    # Save weights to HDF5.
    model.save_weights(model_name + "_weights.h5")
    print("\nSaved model to disk as: {}.json, {}_weights.h5".format(model_name, model_name))
    if history is not None:
        file_name = "lenet_train_hist_dict"
        with open(file_name, 'wb') as f:
            pickle.dump(history.history, f)
        print("\nSaved training history to disk as: {}".format(file_name))


def main():
    """
    LeNet Model
    -------------------------------------
    - Compiles and trains a LeNet-5 neural network on the MNIST data set of gray-scale images.
    - Plots acc/loss vs. epochs.
    - Saves model data.
    """

    # Defining vars
    model_name = "MNIST_LeNet"
    img_x = 28
    img_y = 28
    validation_frac = 12
    u1 = 120
    u2 = 84
    n_classes = 10
    n_epochs = 10

    # Load data.
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Reshaping to make 4D tensor compatible with Keras.

    X_train = X_train.reshape(X_train.shape[0], img_x, img_y, 1)
    X_test = X_test.reshape(X_test.shape[0], img_x, img_y, 1)
    input_shape = (img_x, img_y, 1)

    # Data normalization (needs to be type float for division to work).
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    y_train_1h = to_categorical(y_train, 10)
    y_test_1h = to_categorical(y_test, 10)

    # Creating validation set from training set.
    start_idx = X_train.shape[0] - (X_train.shape[0] // validation_frac)
    X_val = X_train[start_idx::]
    y_val_1h = y_train_1h[start_idx::]
    X_train = X_train[:start_idx]
    y_train_1h = y_train_1h[:start_idx]

    # Create and compile model.
    model = lenet_model(u1, u2, input_shape, n_classes)

    # Train model.
    hist = model.fit(X_train, y_train_1h, batch_size=64, epochs=n_epochs,
                     verbose=1, validation_data=(X_val, y_val_1h))

    # Plot metrics.
    my_plot(hist, model_name)

    # Evaluate model.
    scores = model.evaluate(x=X_test, y=y_test_1h, batch_size=128, verbose=1)
    [print("-", end="") for i in range(20)]
    print("\nMetrics: {}".format(scores))
    [print("-", end="") for i in range(20)]

    # Save model.
    save_model(model, model_name, history=hist)


if __name__ == "__main__":
    main()
