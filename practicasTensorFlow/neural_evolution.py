import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import random
import os
from dotenv import load_dotenv

# Cargamos las variables de entorno
load_dotenv(dotenv_path='./../variables.env')

# set the seed
seed = os.getenv('SEED')
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

# Class that is a representation of a neural network
class NeuralNetwork:
  def __init__(self):
    self.network = []
    self.activations = [
      'relu',
      'sigmoid',
      'tanh',
      'softmax',
      'softplus',
      'softsign',
      'selu',
      'elu',
      'exponential',
      'linear',  # Por defecto, si no se especifica ninguna función de activación, se usa 'linear'
    ]
    self.accuracy = 0
    self.loss = 0
    self.compile_params = {
      'loss': 'sparse_categorical_crossentropy',
      'optimizer': keras.optimizers.Adam(learning_rate=0.001),
      'metrics': ['accuracy']
    }
    self.fit_params = {
      'epochs': 5,
      'batch_size': 32
    }

  def add_layer(self, units:int, activation:str):
    """ 
    Add a layer to the last layer of network 
    units: number of neurons in the layer
    activation: activation function of the layer
    """
    self.network.append([units, activation])

  def random_initialization(self, max_layers:int, max_units:int):
    """
    Random initialization of the network
    max_layers: maximum number of layers
    max_units: maximum number of neurons in a layer
    """
    n_layers = np.random.randint(1, max_layers)
    for _ in range(n_layers):
      self.add_layer(np.random.randint(1, max_units), np.random.choice(self.activations))

  def compile(self, X_train:list, y_train:list, X_test:list, y_test:list, classes:int):
    """
    Compile the network
    X_train: training dataset
    y_train: training labels
    X_test: test dataset
    y_test: test labels
    return: loss and accuracy of the model
    """
    self.model = keras.Sequential()
    self.model.add(keras.layers.Flatten(input_shape=X_train.shape[1:]))
    for layer in self.network:
      self.model.add(keras.layers.Dense(layer[0], activation=layer[1], kernel_initializer='ones', bias_initializer='ones'))
    # the last layer must have the same number of classes
    self.model.add(keras.layers.Dense(classes, activation='softmax', kernel_initializer='ones', bias_initializer='ones'))
    # update the optimizer for evolutive algorithms
    self.compile_params['optimizer'] = keras.optimizers.Adam(learning_rate=0.001)
    self.model.compile(**self.compile_params)
    self.model.fit(X_train, y_train, **self.fit_params)
    self.loss, self.accuracy = self.model.evaluate(X_test, y_test)
    return self.loss, self.accuracy
  

# test creating a neural network of three layers whit 50, 100
""" 
from tensorflow.keras.datasets import mnist
nn = NeuralNetwork()

# add layers
nn.add_layer(100, 'relu')
nn.add_layer(100, 'relu')

# load the dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# compile the model
acc, loss = nn.compile(X_train, y_train, X_test, y_test, 10)

print('Accuracy:', acc)
print('Loss:', loss) """

if __name__ == '__main__':
  from tensorflow.keras.datasets import mnist
  nn = NeuralNetwork()

  # add layers
  #nn.add_layer(50, 'tanh')

  # load the dataset
  (X_train, y_train), (X_test, y_test) = mnist.load_data()

  # Normalizar las imágenes a un rango de 0 a 1
  X_train = X_train.astype('float32') / 255.0
  X_test = X_test.astype('float32') / 255.0

  # Si las imágenes son de una sola canal (escala de grises), necesitas agregar una dimensión extra
  # para que sean compatibles con las capas de convolución de Keras
  X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
  X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

  # compile the model
  acc, loss = nn.compile(X_train, y_train, X_test, y_test, 10)

  print('Accuracy:', acc)
  print('Loss:', loss)