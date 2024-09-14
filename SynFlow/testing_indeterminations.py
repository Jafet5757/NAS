import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow import keras
import prettytable as pt
import numpy as np
import random
import os
from dotenv import load_dotenv

# Cargamos las variables de entorno
""" load_dotenv(dotenv_path='C:/Users/Jafet/Documents/Escuela-Estudio/TT/variables.env')
print(os.getenv('SEED'))
# set the seed
seed = int(os.getenv('SEED'))  # Asegúrate de convertir la semilla a entero
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1' """

def linearize(model):
  """ 
    Get the absolute value of the weights of the model and return the signs of the weights 
  """
  signs = []
  for layer in model.layers:
    if len(layer.get_weights()) > 0:
      weights = layer.get_weights()
      layer_signs = [np.sign(w) for w in weights]
      new_weights = [np.abs(w) for w in weights]
      layer.set_weights(new_weights)
      signs.append(layer_signs)
  return signs


def compute_synflow_per_weight(model, input_shape):
  # Añadimos la dimensión del batch en input_shape
  batch_input_shape = (1,) + input_shape
  
  # Genera un input de unos con la forma del batch
  random_input = tf.ones(batch_input_shape)
  
  with tf.GradientTape() as tape:
      # Realiza una pasada hacia adelante para obtener la salida de la red
      output = model(random_input)
      
      # Calcula la suma de los outputs para obtener un escalar
      loss = tf.reduce_sum(output)
  
  # Calcula los gradientes de la "loss" con respecto a los pesos del modelo
  gradients = tape.gradient(loss, model.trainable_variables)

  # Muestra los gradientes
  print("Gradients:")
  print(gradients)
  
  # Calcula los scores (producto elemento a elemento de los pesos y los gradientes)
  scores = [tf.abs(weight * abs(grad)) for weight, grad in zip(model.trainable_variables, gradients)]
  
  return scores

def example():
  conv_args = {
    "activation": "relu",
    "padding": "same",
    "kernel_initializer": "ones",
    "bias_initializer": "ones"
  }
  
  # Modelo con inicialización y activación revisadas
  model_bad = tf.keras.Sequential([
      # Primera capa convolucional
      layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1), **conv_args),
      layers.MaxPooling2D((2, 2)),
      
      # Segunda capa convolucional con reducción de tamaño
      layers.Conv2D(64, (3, 3), **conv_args),
      layers.MaxPooling2D((2, 2)),
      
      # Capa convolucional adicional para reducir el tamaño
      layers.Conv2D(128, (3, 3), **conv_args),
      layers.MaxPooling2D((2, 2)),
      
      # Capa convolucional adicional con reducción
      layers.Conv2D(256, (3, 3), **conv_args),
      
      # Capa convolucional final con reducción significativa
      layers.Conv2D(512, (3, 3), **conv_args),

      # Capa convolucional final con reducción significativa
      layers.Conv2D(10, (3, 3), activation='softmax'),
  ])

  # Compilamos el modelo
  model_bad.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

  # Construimos el modelo
  model_bad.build()

  # Entrenamos una época con datos aleatorios
  # Genera 100 muestras aleatorias con tamaño 28x28 y 1 canal (grayscale)
  X_train = np.random.rand(100, 28, 28, 1)

  # Genera etiquetas aleatorias con enteros de clase
  y_train = np.random.randint(0, 10, (100, 1, 1, 1))

  # Entrenamiento por una época
  model_bad.fit(X_train, y_train, epochs=3)

  # Calcula los SynFlow scores usando el código previamente ajustado
  score_model_bad = compute_synflow_per_weight(model_bad, (28, 28, 1))

  table = pt.PrettyTable()
  table.field_names = ["Layer", "Bad model"]
  for i, score in enumerate(score_model_bad):
    table.add_row([f"Layer {i}", np.sum(score)])
  print(table)

  score_model_bad_sum = np.sum([np.sum(score) for score in score_model_bad])

  print(f"Total synflow score of bad model: {score_model_bad_sum}")
  
  return score_model_bad_sum


# Ejecutamos el ejemplo 1000 veces y obtenemos la tasa de score = 0
table = pt.PrettyTable()
table.field_names = ["Score"]
zros = 0
for i in range(10):
  score = example()
  table.add_row([score])
  if score == 0:
    zros += 1
    
print(table)
print(f"Total number of scores equal to 0: {zros}")

# Después de 5 ejecuciones de 1000 calculos de score la tasas de score = 0 son las siguientes
# 641, 619, 636, 642, 642
# Promedio = 636 (63.6%)
# Desviación estándar = 8.7863
# Varianza = 77.2