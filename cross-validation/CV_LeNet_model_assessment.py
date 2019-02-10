import pickle
from keras.utils import to_categorical
from keras.datasets import mnist
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping
from MNIST_lenet import save_model


def get_model(u1, u2, dim, n_classes):
    from MNIST_lenet import lenet_model as lm
    return lm(u1, u2, dim, n_classes)



def strat_k_fold(n_folds, X, y, u1, u2, dim, n_classes, n_epochs):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)

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
        model = lenet = get_model(u1, u2, dim, n_classes)

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
        model = None
    return trained_models, model_metrics


def main():
    model_name = "MNIST_LeNet"
    img_x = 28
    img_y = 28
    validation_frac = 12
    u1 = 120
    u2 = 84
    n_classes = 10
    n_epochs = 10
    n_folds = 5

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

    mn = "CV_LeNet"
    for i in range(len(models)):
        s = "_{}".format(i)
        save_model(models[i], mn + s)
    with open("res_dict", 'wb') as f:
        pickle.dump(res, f)


if __name__ == "__main__":
    main()

