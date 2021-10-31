from tensorflow import keras, convert_to_tensor

from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array

from keras import layers

import numpy as np
import tensorflow as tf


def detect_squares(image_path, gaussian_blur_n=5,
                   gradient_intensity_relu_threshold=.8,
                   pool_size=8,
                   square_sizes=range(3, 16),
                   relu_transpose_conv_threshold=.26):
    img = load_data(image_path, pool_size)
    grayscale = to_grayscale(img, r=.299, g=.587, b=.114)
    blurred = gaussian_blur(grayscale, gaussian_blur_n)

    gradients = compute_gradient(blurred)
    gradient = gradient_intensity(gradients)
    gradient_intensity_after_relu = gradient_intensity_relu(gradient, gradient_intensity_relu_threshold)
    pooled_gradient = pooling_gradient_intensity(gradient_intensity_after_relu, pool_size)

    y1 = transpose_convolution_after_pooling(pooled_gradient, square_sizes)
    y2 = relu_after_transpose_convolution(y1, relu_transpose_conv_threshold)
    y3 = transpose_convolution_after_relu(y2, square_sizes)

    upsampled = upsample(y3, pool_size)
    reduced = reduce(upsampled)

    img_with_overlay = img + reduced

    return img_with_overlay


def load_data(image_path, pool_size):
    image = load_img(image_path)

    numpy_data = img_to_array(image)

    h, w, k = numpy_data.shape
    hd = pool_size - (h % pool_size)
    wd = pool_size - (w % pool_size)

    if hd != pool_size:
        for _ in range(hd):
            row = np.zeros((1, w, k), dtype=numpy_data.dtype)
            numpy_data = np.concatenate((numpy_data, row), axis=0)

        h, w, k = numpy_data.shape

    if wd != pool_size:
        for _ in range(wd):
            column = np.zeros((h, 1, k), dtype=numpy_data.dtype)
            numpy_data = np.concatenate((numpy_data, column), axis=1)

        h, w, k = numpy_data.shape

    return convert_to_tensor(numpy_data) / 255.0


# Grayscale
def rgb_kernel(r, g, b):
    def init_kernel(shape, dtype=tf.float32):
        # shape = (height x width) x in_channels x out_channels
        red = np.array([[r]])
        green = np.array([[g]])
        blue = np.array([[b]])

        kernel_in = np.array([[red, green, blue]])
        kernel_in = np.transpose(kernel_in, (2, 3, 1, 0))

        return tf.constant(kernel_in, dtype=dtype)

    return init_kernel


def to_grayscale(rgb_image, r, g, b):
    layer = layers.Conv2D(
        1, (1, 1),
        strides=1,
        kernel_initializer=rgb_kernel(r, g, b),
        padding='same',
        input_shape=rgb_image.shape,
        trainable=False,
    )

    return layer(tf.reshape(rgb_image, shape=[1, *rgb_image.shape]))


# Blur
def gaussian_kernel(n, std=1):
    def init_kernel(shape, dtype=tf.float32):
        # shape = (height x width) x in_channels x out_channels
        G = np.zeros((n, n))

        std_2 = std ** 2
        const = 1 / (2 * np.pi * std_2)

        y_center = n // 2
        x_center = n // 2
        for y in range(n):
            dy = y - y_center
            for x in range(n):
                dx = x - x_center

                G[dy][dx] = const * np.exp(-(dx ** 2 + dy ** 2) / (2 * std_2))

        kernel_in = np.array([[G]])
        kernel_in = np.transpose(kernel_in, (2, 3, 1, 0))

        return tf.constant(kernel_in, dtype=dtype)

    return init_kernel


def gaussian_blur(grayscale, n):
    layer = layers.Conv2D(
        1, (n, n),
        strides=1,
        kernel_initializer=gaussian_kernel(n),
        padding='same',
        trainable=False,
    )

    return layer(grayscale)


# Gradient
def sobel_kernel():
    def init_kernel(shape, dtype=tf.float32):
        # shape = (height x width) x in_channels x out_channels
        sobel_x = np.array([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1]
        ])

        sobel_y = np.array([
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1],
        ])

        kernel_in = np.array([[sobel_x, sobel_y]])
        kernel_in = np.transpose(kernel_in, (2, 3, 0, 1))

        return tf.constant(kernel_in, dtype=dtype)

    return init_kernel


def compute_gradient(blurred):
    layer = layers.Conv2D(
        2, (3, 3),
        strides=1,
        kernel_initializer=sobel_kernel(),
        padding='same',
        trainable=False,
    )

    return layer(blurred)


def gradient_intensity(gradients):
    gradients = tf.transpose(gradients, (0, 3, 1, 2))[0]
    gradients_x, gradients_y = tf.pow(gradients[0], 2), tf.pow(gradients[1], 2)
    summed = tf.sqrt(gradients_x + gradients_y)
    return tf.reshape(summed, [1, *summed.shape, 1])


def gradient_intensity_relu(gradient_intensity, threshold):
    relu_layer = keras.layers.ReLU(threshold=threshold)
    return relu_layer(gradient_intensity) / 255.0


def pooling_gradient_intensity(gradient, pool_size):
    max_pool_2d = keras.layers.MaxPooling2D(pool_size=(pool_size, pool_size),
                                            strides=(pool_size, pool_size), padding='same')
    return max_pool_2d(gradient)


def square_kernel(n):
    # shape = (height x width) x in_channels x out_channels
    def make_kernel(shape, dtype=tf.float32):
        square = np.zeros((n, n))
        for i in range(n):
            square[i, 0] = 1.0
            square[i, n - 1] = 1.0
            square[0, i] = 1.0
            square[n - 1, i] = 1.0

        kernel_in = np.array([[square]])
        kernel_in = np.transpose(kernel_in, (2, 3, 1, 0))

        return tf.constant(kernel_in, dtype=dtype)

    return make_kernel


def transpose_convolution_layers(square_sizes):
    for n in square_sizes:
        layer = keras.layers.Conv2DTranspose(1, (n, n), padding='same', kernel_initializer=square_kernel(n))
        yield layer


def transpose_convolution_after_pooling(pooled_gradients, square_sizes):
    layers = transpose_convolution_layers(square_sizes)
    return [layer(pooled_gradients) for layer in layers]


def relu_after_transpose_convolution(ys, threshold=.26):
    threshold_layer = keras.layers.ReLU(threshold=threshold)
    return [threshold_layer(y) for y in ys]


def transpose_convolution_after_relu(ys, square_sizes):
    layers = transpose_convolution_layers(square_sizes)
    return [layer(y) for y, layer in zip(ys, layers)]


def upsample(ys, pool_size):
    upsample_layer = keras.layers.UpSampling2D(size=(pool_size, pool_size))
    return [upsample_layer(y) for y in ys]


def reduce(ys):
    kernel_in = np.array([[[[0, 1, 1]]]])
    kernel = tf.constant(kernel_in, dtype=tf.float32)
    return sum([tf.nn.conv2d(y, kernel, strides=1, padding='VALID') for y in ys])[0]
