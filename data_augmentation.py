import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import math
import time


(X_TRAIN, Y_TRAIN), (X_TEST, Y_TEST) = tf.keras.datasets.cifar10.load_data()
CIFAR10_CLASSES = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']


class Augment_Image():
    def __init__(self, image, size):
        self.image = image
        self.cropped_image = self.center_crop()
        self.size = size
    
    def center_crop(self):
        return tf.image.central_crop(self.image, 0.75)
    
    def next(self):
        image = self.randomly_rotate(self.image)
        # image = self.randomly_adjust_brightness(image)
        image = self.randomly_adjust_saturation(image)
        image = self.randomly_crop(image)
        return image

    def randomly_rotate(self, image):
        data_augmentation = tf.keras.Sequential([
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(0.2)
        ])
    
        image_batch = tf.expand_dims(image, 0)
        augmented_image = data_augmentation(image_batch)
        return augmented_image[0]
    
    def randomly_adjust_brightness(self, image):
        augmented_image = tf.image.stateless_random_brightness(image, 0.2, (1, 2))
        return augmented_image
    
    def randomly_adjust_saturation(self, image):
        augmented_image = tf.image.stateless_random_saturation(image, 0.5, 1.5, (1, 2))
        return augmented_image
    
    def randomly_crop(self, image):
        augmented_image = tf.image.stateless_random_crop(image, self.size, (1,2))
        return augmented_image


def prepare_data(data):
    images = []
    labels = []

    for i in range(len(data[0])):
        image = Augment_Image(data[0][i], (24, 24, 3))
        label = data[1][i]
        images.append(image.cropped_image)
        labels.append(label)

        for j in range(1):
            augmented_image = image.next()
            images.append(augmented_image)
            labels.append(label)
        
        if (i+1) % 10 == 0:
            print(i+1)

    return images, labels


def plot_some_images(start_index):
    x_train = np.load('aug_x_train.npy')
    y_train = np.load('aug_y_train.npy')

    print(len(x_train))

    plt.figure(figsize=(8, 8))
    for i in range(9):
        image = x_train[i+start_index]
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image)
        plt.axis("off")

    plt.show()


def augment_data():
    start = time.time()
    x_train, y_train = prepare_data((X_TRAIN, Y_TRAIN))
    end = time.time()

    print(end-start)

    x_train = tf.convert_to_tensor(x_train).numpy()
    y_train = tf.convert_to_tensor(y_train).numpy()
    
    np.save('aug_x_train', x_train)
    np.save('aug_y_train', y_train)


def main():
    x_train = np.load('aug_x_train.npy')
    y_train = np.load('aug_y_train.npy')

    Display(x_train, y_train)


if __name__ == "__main__":
    main()
