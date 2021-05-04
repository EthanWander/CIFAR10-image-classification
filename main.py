import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import math
from display import Display


(X_TRAIN, Y_TRAIN), (X_TEST, Y_TEST) = tf.keras.datasets.cifar10.load_data()
CIFAR10_CLASSES = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']


def crop_images(x):
    ret = []
    for image in x:
        ret.append(tf.image.central_crop(image, 0.75))
    return ret


def train():
    x_train = np.load('aug_x_train.npy')
    y_train = np.load('aug_y_train.npy')
    
    x_train_norm = keras.utils.normalize(x_train, axis=1)
    y_train_categorical = to_categorical(y_train)

    model = keras.Sequential(
        [
            keras.Input(shape=(24, 24, 3, )),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.Dropout(0.5),
            layers.MaxPooling2D((2,2)),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.Dropout(0.5),
            layers.MaxPooling2D((2,2)),
            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.Dropout(0.5),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.25),
            layers.Dense(64, activation="relu"),
            layers.Dense(10, activation="softmax")
        ]
    )

    opt = keras.optimizers.Adam(lr=0.001)

    model.compile(
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        optimizer=opt
    )

    fit = model.fit(x_train_norm, y_train_categorical, validation_data=(X_TEST_norm, Y_TEST_categorical), batch_size=64, epochs=100, verbose=1)

    model.save('100epochs-1to1aug')


def main():
    X_TEST_norm = keras.utils.normalize(crop_images(X_TEST), axis=1)
    Y_TEST_categorical = to_categorical(Y_TEST)

    model = tf.keras.models.load_model('100epochs-1to1aug')

    predictions = model.predict([X_TEST_norm])

    y_predicted = []
    for prediction in predictions:
        y_predicted.append([np.argmax(prediction)])
    
    print(y_predicted[0])

    Display(X_TEST, y_predicted)


if __name__ == "__main__":
    main()
