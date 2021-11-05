import tensorflow as tf
import keras
import tensorflow.keras.layers as tfl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

# function for showing image in our dataset
def display(X_train, Y_train, CLASS_NAMES=None, no_show = 9):
  fig = plt.figure(figsize=(8, 8))
  n = math.ceil(math.sqrt(no_show))
  for i in range(no_show):  
    fig.add_subplot(n, n, i+1)
    plt.imshow(np.squeeze(X_train[i]), cmap=plt.get_cmap('gray'))
    if CLASS_NAMES:
      plt.title(CLASS_NAMES[Y_train[i][0]])
    plt.axis('off')
  plt.show()
  return

# load mnist dataset
from keras.datasets import mnist
def load_mnist():
  (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

  # Reshape image to (#image,w,h,channel)
  X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
  Y_train = np.asarray(Y_train).astype('float32').reshape((-1,1))

  X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
  Y_test = np.asarray(Y_test).astype('float32').reshape((-1,1))

  X_train = X_train/255.
  X_test = X_test/255.


  # Reserve 10,000 samples for validation
  X_val = X_train[-10000:]
  Y_val = Y_train[-10000:]

  X_train = X_train[:-10000]
  Y_train = Y_train[:-10000]

  return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)


# load cifar10 dataset
from keras.datasets import cifar10
def load_cifar10():
  (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

  X_train = X_train/255.
  X_test = X_test/255.


  # Reserve  samples for validation
  X_val = X_train[-5000:]
  Y_val = Y_train[-5000:]

  X_train = X_train[:-5000]
  Y_train = Y_train[:-5000]

  # class name
  CLASS_NAMES= ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

  return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), CLASS_NAMES