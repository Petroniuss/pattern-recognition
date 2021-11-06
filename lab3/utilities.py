import tensorflow as tf
import numpy as np

from tensorflow import keras
from keras import layers
from keras import activations

import matplotlib.pyplot as plt

classes_no = 10


def one_hot_encoding(labels):
    one_hot = tf.one_hot(labels, classes_no, dtype=tf.float32)
    return tf.reshape(one_hot, (labels.shape[0], classes_no))


def plot_accuracy(history):
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    ax = plt.gca()
    ax.set_ylim([0.0, 1.0])
    plt.show()


def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    ax = plt.gca()
    ax.set_ylim([0.0, 2.5])
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


def retrieve_history(name):
    np.load(f'history/{name}.npy', allow_pickle='TRUE').item()


def save_history(name, history):
    np.save(f'history/{name}.npy', history.history)
