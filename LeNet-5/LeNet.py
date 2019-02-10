import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.datasets import mnist
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import model_from_json


def load_model(model_file_name, weights_file_name):
    """
    :param model_file_name: name of saved model without file extension.
    :param weights_file_name: name of saved weights without file extension.
    :return: keras.Sequensial uncompiled model.
    -------------------------------------
    Loads model and weights from specified files.
    -------------------------------------
    """
    json_file = open(model_file_name + ".json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weights_file_name + ".h5")
    print("Loaded model from disk")
    return loaded_model


def load_training_data(file_name):
    """
    :param file_name: name of saved training data (pickle)
    :return: training data (dict)
    -------------------------------------
    Loads training data metrics and returns
    a dictionary.
    -------------------------------------
    """
    try:
        with open(file_name, 'rb') as f:
            hist = pickle.load(f)
        return hist
    except:
        print("Could not load file ...")
        return None


def main():
    """
    LeNet model:
    -------------------------------------
    - Evaluates loaded model on MNIST test data.
    - Shows loss and accuracy.
    - Predicts MNIST test data
    - Produces classification report and confusion matrix.
    """

    # Defining variables.
    img_x = 28
    img_y = 28
    model_name = "MNIST_LeNet"
    model_weights = "MNIST_LeNet_weights"

    # Load data and prepare.
    (_X_train, _y_train), (X_test, y_test) = mnist.load_data()
    X_test = X_test.reshape(X_test.shape[0], img_x, img_y, 1)
    input_shape = (img_x, img_y, 1)

    # Data normalization (needs to be type float for division to work).
    X_test = X_test.astype('float32')
    X_test /= 255
    y_test_1h = to_categorical(y_test, 10)

    # Load model and compile.
    model = load_model(model_name, model_weights)
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=SGD())

    # Evaluate model.
    scores = model.evaluate(x=X_test, y=y_test_1h, batch_size=128, verbose=1)
    [print("-", end="") for i in range(20)]
    print("\nMetrics: {}".format(scores))
    [print("-", end="") for i in range(20)]

    # Predict test set.
    predictions = model.predict(X_test)

    # Reverting one hot encoding.
    predictions = [np.argmax(i) for i in predictions]

    # Displaying report.
    [print("=", end="") for i in range(20)]
    print("\nConfution Matrix:")
    [print("-", end="") for i in range(20)]
    print("\n", confusion_matrix(y_test, predictions))
    [print("=", end="") for i in range(20)]
    print("\nCls Report:")
    [print("-", end="") for i in range(20)]
    print("\n", classification_report(y_test, predictions))
    [print("-", end="") for i in range(20)]

    ################################################
    # Uncomment if you want to run the code below. #
    ################################################

    # Load up training history.
    hist = load_training_data("lenet_train_hist_dict")
    print(hist.keys())

    # Print model summary/architecture.
    # print(model.summary())


if __name__ == "__main__":
    main()
