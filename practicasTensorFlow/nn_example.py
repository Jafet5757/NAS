import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from dotenv import load_dotenv

# Cargamos las variables de entorno
load_dotenv(dotenv_path='./../variables.env')

# set the seed
seed = os.getenv('SEED')
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# Load the Fashion MNIST dataset
(X, y), (X_test, y_test) = fashion_mnist.load_data()

# labels for the Fashion MNIST dataset
labels = ["T-shirt/top",
          "Trouser",
          "Pullover",
          "Dress",
          "Coat",
          "Sandal",
          "Shirt",
          "Sneaker",
          "Bag",
          "Ankle boot"]


# show some images
plt.figure(figsize=(14, 8))
ind = np.random.choice(X.shape[0], 20)
for i, img in enumerate(ind):
  plt.subplot(5, 10, i+1)
  plt.title(labels[y[img]])
  plt.imshow(X[img], cmap='gray')
  plt.axis('off')
plt.show()

# preparando el dataset
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
print('Imagenes de entrenamiento: ', X_train.shape)
print('Imagenes de prueba: ', X_test.shape)

# Modelo de la red neuronal
model = keras.Sequential([
  keras.layers.Flatten(input_shape=(28, 28)),
  keras.layers.Dense(50, activation='tanh', kernel_initializer='ones', bias_initializer='ones'),
  keras.layers.Dense(54, activation='tanh', kernel_initializer='ones', bias_initializer='ones'),
  keras.layers.Dense(10, activation='softmax', kernel_initializer='ones', bias_initializer='ones')
])

# Compilado del modelo
model.compile(
  loss="sparse_categorical_crossentropy",
  optimizer=keras.optimizers.Adam(learning_rate=0.001),
  metrics=["accuracy"]
)

# definimos los parametros de entrenamiento
params = {
  'validation_data': (X_test, y_test),
  'epochs': 5,
  'batch_size': 32
}

# Entrenamiento del modelo
history = model.fit(X_train, y_train, **params)

# Evaluacion del modelo
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Precisión en el conjunto de prueba:', test_acc)
print('Perdida en el conjunto de prueba:', test_loss)