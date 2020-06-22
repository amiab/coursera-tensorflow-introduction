# Programming Assignment: Exercise 3 (Improve MNIST with convolutions))

# TO DO: improve MNIST to 99.8% accuracy or more using only a single convolutional layer and a single MaxPooling 2D.
# Stop training once the accuracy goes above this amount. once 99.8% is hit, you should print out the string
# "Reached 99.8% accuracy so cancelling training!" It should happen in less than 20 epochs, so it's ok to hard
# code the number of epochs for training, but your training must end once it hits the above metric.

import tensorflow as tf
from os import path, getcwd, chdir

# Callback class could live in seperate file
class modelfit_callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if( logs.get('accuracy') > 0.998):
            # or check the loss ... if (logs.get('loss') < 0.01):
            print("\nReached 99.8% accuracy so cancelling training!")
            self.model.stop_training = True

# grab mnist.npz from the Coursera Jupyter Notebook
# path = f"{getcwd()}/../tmp2/mnist.npz"
path = f"{getcwd()}\data\mnist.npz"

# Tensorflow v1.x
# config = tf.ConfigProto()
# config = tf.compat.v1.ConfigProto
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)

# GPU not available on my local machine
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# limit  GPU’s memory usage up to 1024MB
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     tf.config.experimental.set_virtual_device_configuration(gpus[0],
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
#   except RuntimeError as e:
#     print(e)

def train_mnist_conv():

    mnist = tf.keras.datasets.mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data(path=path)

    training_images = training_images.reshape(60000, 28, 28, 1)
    training_images = training_images / 255.0;
    test_images = test_images.reshape(10000, 28, 28, 1)
    test_images = test_images / 255.0;

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # test with extra 2 layers
        # tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        # tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # check summary od the convolution
    model.summary()

    mf_callbacks = modelfit_callback()

    # model fitting
    history = model.fit(
        training_images, training_labels, epochs=20, callbacks=[mf_callbacks]
    )
    # evaluate
    test_loss = model.evaluate(test_images, test_labels)

    return history.epoch, history.history['accuracy'][-1]


# run the training
_, _ = train_mnist_conv()