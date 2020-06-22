# Programming Assignment: Exercise 2 (Handwriting Recognition)

# classification exercise using MNIST which has items of handwriting -- the digits 0 through 9.
# similar to Fashion MNIST data set containing items of clothing.
#
# TO DO: Write an MNIST classifier that trains to 99% accuracy or above, and does it without a fixed number of epochs
# -- i.e. stop training once we reach that level of accuracy.
#
# Some notes:
# It should succeed in less than 10 epochs, so it is okay to change epochs= to 10, but nothing larger
# When it reaches 99% or greater it should print out the string "Reached 99% accuracy so cancelling training!"

import tensorflow as tf
from os import path, getcwd, chdir

# Callback class could live in seperate file
class modelfit_callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if( logs.get('accuracy') > 0.99):
            # or check the loss ... if (logs.get('loss') < 0.01):
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True

# grab mnist.npz from the Coursera Jupyter Notebook
# path = f"{getcwd()}/../tmp2/mnist.npz"
path = f"{getcwd()}/data/mnist.npz"

# in google colab we dont need to specify the location
# the data is automatically loaded from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz

def train_mnist():

    mf_callbacks = modelfit_callback()

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data(path=path)
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    # or both in 1 line
    # x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([

        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)

    ])

    # 'accuracy' is the metric label that's what will be shown in the epoch log on the right together with the loss metric
    # make sure the same variable name 'accuracy' is used in the callback class on_epoch_end and in the compile
    # as well as the history[]
    # this is the metric label that will be shown in the epoch log on the right with the loss metric
    # for example if you use 'acc' instead in the callback class then change the variable here too and history['acc']

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # model fitting
    history = model.fit(
        x_train, y_train, epochs=10, callbacks=[mf_callbacks]
    )
    # model fitting
    return history.epoch, history.history['accuracy'][-1]

# run the training
train_mnist()

