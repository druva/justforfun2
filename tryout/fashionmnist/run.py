import tensorflow as tf
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

import math
import numpy as np
import matplotlib.pyplot as plt

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

def initDataSet(verbose):
    #load data set and test set
    dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
    train_dataset, test_dataset = dataset['train'], dataset['test']
    class_names = metadata.features['label'].names
    num_train_examples = metadata.splits['train'].num_examples
    num_test_examples = metadata.splits['test'].num_examples
    if verbose:
        print("Class names: {}".format(class_names))
        print("Number of training examples: {}".format(num_train_examples))
        print("Number of test examples:     {}".format(num_test_examples))
    train_dataset, test_dataset = doNormalize(train_dataset, test_dataset, verbose)
    return dataset, metadata, train_dataset, test_dataset, class_names, num_train_examples, num_test_examples

def normalize(images, labels):
  # print("A", images)
  images = tf.cast(images, tf.float32)
  # print("B", images)
  images /= 255
  # print("C", images)
  return images, labels

def doNormalize(train_dataset, test_dataset, verbose):
    if verbose:
        print("train_dataset-Before", train_dataset)
        print("test_dataset-Before", test_dataset)

    train_dataset =  train_dataset.map(normalize)
    test_dataset  =  test_dataset.map(normalize)

    if verbose:
        print("train_dataset-After", train_dataset)
        print("test_dataset-After", test_dataset)
    return train_dataset, test_dataset

def viewOneImage(dataset, verbose):
    if verbose:
        for image, label in dataset.take(1):
            break;
        image = image.numpy().reshape((28,28))

        plt.figure()
        plt.imshow(image, cmap=plt.cm.binary)
        plt.colorbar()
        plt.grid(False)
        plt.show()

def viewMoreImages(dataset, class_names, verbose):
    if verbose:
        plt.figure(figsize=(10,10))
        for i, (image, label) in enumerate(dataset.take(25)):
            image = image.numpy().reshape((28,28))
            plt.subplot(5,5,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(image, cmap=plt.cm.binary)
            plt.xlabel(class_names[label])
        plt.show()

def createModel():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    return model

def compileModel(model):
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
    return model

def run():
    verbose = True
    dataset, metadata, train_dataset, test_dataset, class_names, num_train_examples, num_test_examples = initDataSet(verbose)
    # viewOneImage(test_dataset, verbose)
    # viewMoreImages(test_dataset, class_names, verbose)
    model = createModel()
    model = compileModel(model)

run()