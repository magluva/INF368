import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from keras.datasets import mnist
from sklearn.metrics import classification_report, confusion_matrix

def simple_model(input_nodes, dim, n_classes, dm=True):
    """
    @params: input_dim (int), dim (int), n_classes (int), dm (boolean)
        - dm (boolean) = True ---> Displays model architecture.
    Creates a two-layered simple network:
    -------------------------------------
    Layer 1:      Dense, Fully connected input layer
    Activation 1: Sigmoid
    Layer 2:      Dense, fully connected output layer
    Activation 2: Softmax
    -------------------------------------
    """
    model = Sequential()
    model.add(Dense(input_nodes, activation="sigmoid", input_shape=dim))
    model.add(Dense(n_classes, activation="softmax"))
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=RMSprop())
    if dm:
        model.summary()
    return model

def my_plot(history):
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
    return plt.show()

def main():
    # Defining vars
    img_x = 28
    img_y = 28
    validation_frac = 12
    n_classes = 10
    n_epochs = 10

    # Load data.
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Reshaping to make 4D tensor compatible with Keras.

    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    input_shape = (img_x * img_y, )

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
    model = simple_model(32, input_shape, n_classes)

    # Train model.
    hist = model.fit(X_train, y_train_1h, batch_size=64, epochs=n_epochs,
                     verbose=1, validation_data=(X_val, y_val_1h))

    # Plot metrics.
    my_plot(hist)

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



if __name__ == "__main__":
    main()
