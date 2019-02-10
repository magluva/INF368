import pickle
from keras.models import Sequential
from keras.layers import AveragePooling2D, Conv2D, Dense, Flatten
from keras.utils import to_categorical
from keras.datasets import mnist
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping
from MNIST_lenet import save_model
from keras.optimizers import SGD


def lenet_model(u1, u2, dim, n_classes):
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
    return model


def CV_sdg():
    sgd = dict()
    sgd["no_momentum"] = SGD(momentum=0.0, nesterov=False)
    sgd["momentum"] = SGD(momentum=0.9, nesterov=False)
    sgd["nesterov"] = SGD(momentum=0.9, nesterov=True)
    return sgd


def strat_k_fold(n_folds, X, y, u1, u2, dim, n_classes, n_epochs):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
    CV_sdg_dict = CV_sdg()
    CV_sgd_keys = ["no_momentum", "momentum", "nesterov"]

    X_tr, X_te = X
    y_tr, y_te = y

    trained_models = list()
    model_metrics = dict()

    for i, (train, val) in enumerate(skf.split(X_tr, y_tr)):
        # Print info.
        print("Fold:    {}/{}".format((i+1), n_folds))

        # Splitting data.
        X_train = X_tr[train]
        y_train = y_tr[train]
        X_val = X_tr[val]
        y_val = y_tr[val]
        y_train_1h = to_categorical(y_train, n_classes)
        y_val_1h = to_categorical(y_val, n_classes)
        y_test_h = to_categorical(y_te, n_classes)

        # Compiling model.
        model = lenet_model(u1, u2, dim, n_classes)
        model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=CV_sdg_dict[CV_sgd_keys[i]])

        # Defining callback.
        callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=0)]

        # Training model.
        history = model.fit(X_train, y_train_1h, batch_size=64, epochs=n_epochs,
                            shuffle=True, callbacks=callbacks, verbose=1, validation_data=(X_val, y_val_1h))

        # Evaluate model.
        res = model.evaluate(x=X_te, y=y_test_h, batch_size=128, verbose=1)

        # store models.
        trained_models.append(model)

        # Store results.
        model_metrics[i] = res, history.history["acc"], history.history["val_acc"],\
                           history.history["loss"], history.history["val_loss"]
    return trained_models, model_metrics


def main():
    img_x = 28
    img_y = 28
    u1 = 120
    u2 = 84
    n_classes = 10
    n_epochs = 10
    n_folds = 3

    #Load data.
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Reshaping to make 4D tensor compatible with Keras.

    X_train = X_train.reshape(X_train.shape[0], img_x, img_y, 1)
    X_test = X_test.reshape(X_test.shape[0], img_x, img_y, 1)
    dim = (img_x, img_y, 1)

    # Data normalization (needs to be type float for division to work).
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    X = (X_train, X_test)
    y = (y_train, y_test)
    models, res = strat_k_fold(n_folds, X, y, u1, u2, dim, n_classes, n_epochs)

    mn = "CV_solver_LeNet"
    for i in range(len(models)):
        s = "_{}".format(i)
        save_model(models[i], mn + s)
    with open("res_dict", 'wb') as f:
        pickle.dump(res, f)


if __name__ == "__main__":
    main()

