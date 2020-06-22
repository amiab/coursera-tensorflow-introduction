# Programming Assignment: Exercise 4 (Handling complex images)

# create image classifier for complex images, Dataset  contains 80 images, 40 happy and 40 sad.
# Create a convolutional neural network that trains to 100% accuracy on these images,
# Use a callback to cancel training once accuracy is greater than .999.
#
# Hint -- it will work best with 3 convolutional layers.

import tensorflow as tf
import os
import zipfile as zip
from os import path, getcwd, chdir
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DESIRED_ACCURACY = 0.999
ROOT_DIR = getcwd()

# callback to cancel training once accuracy is greater than .999.
class modelfit_callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if( logs.get('accuracy') > DESIRED_ACCURACY):
            print("\nReached desired accuracy so cancelling training!")
            self.model.stop_training = True


path = f"{ROOT_DIR}/data/happy-or-sad.zip"

zip_ref = zip.ZipFile(path, 'r')
zip_ref.extractall(f"{ROOT_DIR}/data/h-or-s")
zip_ref.close()


def train_happy_sad_model():

    mf_callbacks = modelfit_callback()

    # Define and Compile the Model.images are 150 X 150
    model = tf.keras.models.Sequential([
        # input shape with size of the image 150x150 with 3 bytes color
        # 1st convolution
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # 2nd convolution
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # 3rd convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # 4th convolution
        # tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        # tf.keras.layers.MaxPooling2D(2,2),
        # 5th convolution
        # tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        # tf.keras.layers.MaxPooling2D(2,2),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),
        # Only 1 output neuron. It will contain a value from 0-1 where 0 for class ('sad') and 1 for class ('happy')
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        loss='binary_crossentropy',
        # larning rate
        optimizer=RMSprop(lr=0.001),
        metrics=['accuracy']
    )

    # create an instance of an ImageDataGenerator called train_datagen
    # And a train_generator by calling train_datagen.flow_from_directory

    # All images will be rescaled by 1./255
    train_datagen = ImageDataGenerator(rescale=1 / 255)

    # use a target_size of 150 X 150.
    # Flow training images in batches of 128 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(
        f'{ROOT_DIR}/data/h-or-s',  # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=128,
        class_mode='binary' # Since we use binary_crossentropy loss, we need binary labels
    )

    # Expected output: 'Found 80 images belonging to 2 classes'

    # call model.fit and train for a number of epochs.
    # model.fit_generator is deprecated . model.fit supports generator
    history = model.fit(
        train_generator,
        #steps_per_epoch=8,
        epochs=15,
        verbose=1,
        callbacks=[mf_callbacks]
    )

    return history.history['accuracy'][-1]

# run the training
train_happy_sad_model()
