import os
import numpy as np
#import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical

def mk_data(validation=None, log=False):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    if log:
        # saves imege of first 3 digits and their pixel distributions.
        check_digits(X_train, y_train, 3)
        # We get min value of 0 and max value of 255 (expected)
        # Get data shapes.
        shape_info = get_shapes(X_train, y_train, X_test, y_test)
        # Write shapes to log file.
        path = "logfile.txt"
        my_log(path, "Shapes:\n------------------------------\n")
        my_log(path, "X_train: {}\n".format(shape_info["X_train"]))
        my_log(path, "y_train: {}\n".format(shape_info["y_train"]))
        my_log(path, "X_test: {}\n".format(shape_info["X_test"]))
        my_log(path, "y_test: {}\n".format(shape_info["y_test"]))
        my_log(path, "-------------------------------\n")
        
    # Reshaping to make 4D tensor compatible with Keras.
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)
    # Data normalization (needs to be type float for division to work).
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    if log:
        # Get new shape data.
        new_shape = get_shapes(X_train, y_train, X_test, y_test)
        # Log new shape data.
        my_log(path, "New Shapes:\n------------------------------\n")
        my_log(path, "X_train: {}\n".format(new_shape["X_train"]))
        my_log(path, "y_train: {}\n".format(new_shape["y_train"]))
        my_log(path, "X_test: {}\n".format(new_shape["X_test"]))
        my_log(path, "y_test: {}\n".format(new_shape["y_test"]))
        my_log(path, "-------------------------------\n")

    # Validation sets
    if validation is None:
        X_val = X_test
        y_val = y_test
    else:
        start_idx = X_train.shape[0] - (X_train.shape[0]//validation)
        X_val = X_train[start_idx::]
        y_val = y_train[start_idx::]
        X_train = X_train[:start_idx]
        y_train = y_train[:start_idx]
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    

def my_log(path, msg):
    """
    Creates and appends specified data to a log file
    for use when writing the report.
    """
    # Opens and writes specified file.
    # Appends if "f" exists, creates it if not. 
    with open(path, "a") as f:
        f.write(msg)
        # "Touches" the file.
        os.utime(path, None)
    return


def get_shapes(X_train, y_train, X_test, y_test):
    """
    Creates a dict of data subset shapes.
    Just for checking.
    """
    # Creates empty dict
    shapes_dict = dict()
    # Adds shape data as values to the following keys.
    shapes_dict["X_train"] = X_train.shape
    shapes_dict["y_train"] = y_train.shape
    shapes_dict["X_test"] = X_test.shape
    shapes_dict["y_test"] = y_test.shape
    return shapes_dict

def check_digits(X_train, y_train, n):
    """
    Creates a plot of the n first digits, corresponding
    labels, as well as pixel values and saves it to working directory.
    Just for verification purposes.
    """
    # Creates figure and n subplots. 
    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(10, 10))
    axes[0, 1].set_title("Pixel Distribution", fontsize=9)
    for i in range(n):
        # Sets digit ylabel as class label
        axes[i, 0].set_ylabel("Digit: {}".format(y_train[i]), fontsize=7)
        # Plots gray scale images
        axes[i, 0].imshow(X_train[i], cmap="gray")
        # Disable ticks
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        axes[i, 1].set_xlabel("value", fontsize=9)
        axes[i, 1].set_ylabel("pixel", fontsize=9)
        # Plot histogram of pixel value distribtions on the axes
        # to the right of every gray scale image.
        axes[i, 1].hist(X_train[0].reshape(784))
    # Uncomment to show plot at runtime.
    # plt.show() 
    return plt.savefig("digit_samples.png")
