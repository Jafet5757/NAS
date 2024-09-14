import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow import keras
import prettytable as pt
import numpy as np

class ConvolutionalNeuralNetwork:
  def __init__(self, epochs:int, batch_size:int):
    self.network = []
    self.activations = [
      'relu', # Default
      'sigmoid',
      'tanh',
      'softmax',
      'softplus',
      'softsign',
      'selu',
      'elu',
      'exponential',
      'linear'
    ]
    self.layers_types = [
      'Conv2D',
      'MaxPooling2D',
      'Conv2DTranspose',
      'concatenate'
    ]
    self.compile_params = {
      'loss': 'sparse_categorical_crossentropy',
      'optimizer': tf.keras.optimizers.Adam(learning_rate=0.001),
      'metrics': ['accuracy']
    }
    self.fit_params = {
      'epochs': epochs,
      'batch_size': batch_size
    }
    self.accuracy = 0
    self.loss = 0

  def add_layer(self, filters:int=32, kernel_size:tuple=(3, 3), activation:str="relu", type:str="Conv2D", padding:str="same", strides:tuple=(1, 1)):
    """
    Add a layer to the last layer of network 
    filters: number of filters in the layer
    kernel_size: size of the kernel
    activation: activation function of the layer
    type: type of layer (Conv2D, MaxPooling2D, Flatten, Dense)
    padding: padding of the layer
    strides: strides of the layer
    """
    self.network.append({"filters":filters, "kernel_size":kernel_size, "activation":activation, "type":type, "padding":padding, "strides":strides})

  def random_model(self, max_layers:int, max_filters:int, max_kernel_size:int):
    """
    Random initialization of the network, use stride = MaxPooling2D and NO use concatenate
    max_layers: maximum number of layers
    max_filters: maximum number of filters in a layer
    max_kernel_size: maximum size of the kernel
    """
    # Generamos un número aleatorio de capas
    n_layers = np.random.randint(1, max_layers)
    # Si el numero es impar, suma 1
    if n_layers % 2 != 0:
      n_layers += 1

    for i in range(int(n_layers/2)):
      # Generamos la misma cantidad de convoluciones, deben ser simetricas y cunplir la misma cantidad de strides y pooling
      # Generamos la primer mitad de la U-net
      filters = np.random.randint(1, max_filters)
      kernel_size_number = np.random.randint(1, max_kernel_size)
      kernel_size = (kernel_size_number, kernel_size_number)
      activation = np.random.choice(self.activations)
      type = "Conv2D" # Por defecto
      padding = "same" # Por defecto
      strides = (2, 2) if i % 2 == 0 else (1, 1) # Si es par, hacemos un MaxPooling2D, si no, hacemos un Conv2D

      self.network.append({"filters":filters, "kernel_size":kernel_size, "activation":activation, "type":type, "padding":padding, "strides":strides})

    # Generamos la segunda mitad de la U-net (deconvolucion)
    for i in range(int(n_layers/2)):
      filters = np.random.randint(1, max_filters)
      kernel_size_number = np.random.randint(1, max_kernel_size)
      kernel_size = (kernel_size_number, kernel_size_number)
      activation = np.random.choice(self.activations)
      type = "Conv2DTranspose"
      padding = "same" # Por defecto
      strides = (2, 2) if i % 2 == 0 else (1, 1) # Si es par, hacemos un MaxPooling2D, si no, hacemos un Conv2DTranspose

      self.network.append({"filters":filters, "kernel_size":kernel_size, "activation":activation, "type":type, "padding":padding, "strides":strides})

      

  def compile(self, X_train:list, y_train:list, X_test:list, y_test:list, classes:int, bottleneck_layers:int=2, skip:int=2):
    """
    Compile the network
    X_train: training dataset
    y_train: training labels
    X_test: test dataset
    y_test: test labels
    classes: number of classes
    bottleneck_layers: number of layers to skip in the bottleneck
    skip: number of layers to skip in the concatenate
    return: loss and accuracy of the model
    """
    # obtenemos el input_shape
    input_shape = X_train.shape[1:]

    inputs = layers.Input(shape=input_shape)
    layers_sequential = []
    convolutions = [] # Pila de convoluciones para concatenarlas al final
    bottleneck_skiped = False
    deconvolution_stage = False

    # Antes de compilar aplicamos la reparación genética
    self.genetic_reparation(input_shape)

    # Recorremos las capas de la red neuronal
    for (i, layer) in enumerate(self.network):
      if layer["type"] == "Conv2D":
        if i == 0:
          x = layers.Conv2D(layer["filters"], layer["kernel_size"], activation=layer["activation"], padding=layer["padding"], strides=layer["strides"])(inputs)
        else:
          x = layers.Conv2D(layer["filters"], layer["kernel_size"], activation=layer["activation"], padding=layer["padding"], strides=layer["strides"])(layers_sequential[-1])
        layers_sequential.append(x)
        
        if not deconvolution_stage:
          convolutions.append(x)

      elif layer["type"] == "MaxPooling2D":
        # Añadimos una capa de MaxPooling2D de 2x2
        x = layers.MaxPooling2D((2, 2))(layers_sequential[-1])
        layers_sequential.append(x)

      elif layer["type"] == "Conv2DTranspose":
        # Inician las capas de deconvolución
        deconvolution_stage = True
        # Añadimos una capa de Conv2DTranspose
        x = layers.Conv2DTranspose(layer["filters"], layer["kernel_size"], activation=layer["activation"], padding=layer["padding"], strides=layer["strides"])(layers_sequential[-1])
        layers_sequential.append(x)

      elif layer["type"] == "concatenate":
        # Si no se ha saltado la capa bottleneck, saltamos las n siguientes
        if not bottleneck_skiped:
          for _ in range(bottleneck_layers):
            convolutions.pop()
          bottleneck_skiped = True
        # Concatenamos la última capa con la última convolución
        x = layers.concatenate([layers_sequential[-1], convolutions.pop()])
        # Añadimos la capa concatenada
        layers_sequential.append(x)
        # Saltamos las siguientes n-1 capas de skip
        for _ in range(skip-1):
          convolutions.pop()

    outputs = layers.Conv2D(classes, (1, 1), activation='softmax')(layers_sequential[-1])

    model = models.Model(inputs=inputs, outputs=outputs)
    
    self.compile_params['optimizer'] = keras.optimizers.Adam(learning_rate=0.001)
    # Compilamos el modelo
    model.compile(**self.compile_params)

    try:
      # Mostramos el resumen del modelo
      print(model.summary())

      # Entrenamos el modelo
      model.fit(X_train, y_train, **self.fit_params)

      # Evaluamos el modelo
      loss, accuracy = model.evaluate(X_test, y_test)

      # Guardamos la precisión y la pérdida
      self.accuracy = accuracy
      self.loss = loss

      print('Accuracy:', accuracy)
      print('Loss:', loss)
    except Exception as e:
      # Imprimimos la red neuronal
      print(self)
      # Imprimimos el error
      print(e)
      # Lanzamos una excepción si hay un error
      raise Exception("Error al compilar el modelo")
    
    # Liberamos la memoria de la GPU
    tf.keras.backend.clear_session()

    return loss, accuracy
  

  def genetic_reparation(self, input_shape:tuple):
    """ 
    Reparación de la red neuronal, se encarga de reparar la red neuronal si hay un error
    input_shape: forma de la entrada de la red neuronal
    - Si la red no tiene de salida la misma forma (shape) que la entrada, se añade una capa de Conv2D con la misma forma 
    """
    # Obtenemos la forma de la salida de la red neuronal
    output_shape = self.network[-1]["filters"]
    # Si la forma de la salida no es la misma que la de la entrada, añadimos una capa de Conv2D
    if output_shape != input_shape:
      self.add_layer(input_shape[0], (3, 3), 'relu', 'Conv2D')
    return self
  
  def __str__(self):
    table = pt.PrettyTable()
    table.field_names = ["Layer", "Filters", "Kernel Size", "Activation", "Type", "Padding", "Strides"]
    for (i, layer) in enumerate(self.network):
      table.add_row([i, layer["filters"], layer["kernel_size"], layer["activation"], layer["type"], layer["padding"], layer["strides"]])
    return str(table)  

if __name__ == '__main__':
  # probamos a crear una red neuronal convolucional
  from tensorflow.keras.datasets import mnist

  # Importamos el dataset
  (X_train, y_train), (X_test, y_test) = mnist.load_data()

  # Normalizar las imágenes a un rango de 0 a 1
  X_train = X_train.astype('float32') / 255.0
  X_test = X_test.astype('float32') / 255.0

  # Si las imágenes son de una sola canal (escala de grises), necesitas agregar una dimensión extra
  # para que sean compatibles con las capas de convolución de Keras
  X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
  X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

  # Creamos la red neuronal
  cnn = ConvolutionalNeuralNetwork(5, 64)

  # Añadimos las capas
  cnn.add_layer(32, (3, 3), 'relu', 'Conv2D')
  cnn.add_layer(32, (3, 3), 'relu', 'Conv2D')
  cnn.add_layer(type='MaxPooling2D')

  cnn.add_layer(64, (3, 3), 'relu', 'Conv2D')
  cnn.add_layer(64, (3, 3), 'relu', 'Conv2D')
  cnn.add_layer(type='MaxPooling2D')

  cnn.add_layer(128, (3, 3), 'relu', 'Conv2D')
  cnn.add_layer(128, (3, 3), 'relu', 'Conv2D')
  
  cnn.add_layer(64, (3, 3), 'relu', 'Conv2DTranspose', strides=(2, 2))
  cnn.add_layer(type='concatenate')
  cnn.add_layer(64, (3, 3), 'relu', 'Conv2D')
  cnn.add_layer(64, (3, 3), 'relu', 'Conv2D')

  cnn.add_layer(32, (3, 3), 'relu', 'Conv2DTranspose', strides=(2, 2))
  cnn.add_layer(type='concatenate')
  cnn.add_layer(32, (3, 3), 'relu', 'Conv2D')
  cnn.add_layer(32, (3, 3), 'relu', 'Conv2D')
  
  print(cnn)

  # Compilamos la red neuronal
  loss, acc = cnn.compile(X_train, X_train, X_test, X_test, 2, 2, 2)
  