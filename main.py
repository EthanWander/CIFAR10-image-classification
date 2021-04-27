import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import math
import random
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


(X_TRAIN, Y_TRAIN), (X_TEST, Y_TEST) = tf.keras.datasets.cifar10.load_data()
CIFAR10_CLASSES = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']


class Display:
    def __init__(self):
        self.index = self._gen_rand_index()
        self.im = plt.imshow(X_TRAIN[self.index])
        self.title_text = self._get_title_from_index()
        self.title_obj = plt.title(self.title_text)
        
        axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        self.next_button = Button(axnext, 'Next')
        self.next_button.on_clicked(self.next_btn_clicked)

        plt.show()  
            
    def _gen_rand_index(self):
        return int(random.random()*50000)

    def _get_title_from_index(self):
        title_index = Y_TRAIN[self.index][0]
        title_text = CIFAR10_CLASSES[title_index]
        return title_text

    def next_btn_clicked(self, event):
        self.index = self._gen_rand_index()
        self.im.set_data(X_TRAIN[self.index])
        self.title_text = self._get_title_from_index()
        self.title_obj.set_text(self.title_text)


def main():
    # display = Display()
    # X_TRAIN_gray = tf.image.rgb_to_grayscale(X_TRAIN)
    # X_TEST_gray = tf.image.rgb_to_grayscale(X_TRAIN)
    X_TRAIN_norm = keras.utils.normalize(X_TRAIN, axis=1)
    X_TEST_norm = keras.utils.normalize(X_TEST, axis=1)
    Y_TRAIN_categorical = to_categorical(Y_TRAIN)
    Y_TEST_categorical = to_categorical(Y_TEST)

    model = keras.Sequential(
        [
            keras.Input(shape=(32, 32, 3, )),
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

    fit = model.fit(X_TRAIN_norm, Y_TRAIN_categorical, validation_data=(X_TEST_norm, Y_TEST_categorical), batch_size=64, epochs=15, verbose=1)

    # predictions = model.predict([X_TEST_norm])

    # print("\n\n")
    # print("Choose pictures out of the 10,000 available\n")

    # while True:
    #     print("Enter an arbitrary number between 0 and 9999: ")
    #     img_index = input()
    #     img_index = int(img_index)
    #     print("\n")
    #     print(get_img_ascii(X_TEST[img_index]))
    #     print(np.argmax(predictions[img_index]))
    #     print("\n\n")


if __name__ == "__main__":
    main()
