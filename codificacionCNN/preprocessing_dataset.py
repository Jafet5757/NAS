import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Cargar el conjunto de datos Oxford-IIIT Pet Dataset
dataset, info = tfds.load('oxford_iiit_pet', with_info=True)

# Dividir el conjunto de datos en entrenamiento y prueba
train_dataset = dataset['train']
test_dataset = dataset['test']

# Definir el tamaño de las imágenes
IMG_SIZE = 128  # Cambia este tamaño según tus necesidades

# Función para normalizar la imagen, redimensionar y convertir la máscara a one-hot encoding
def process_data(data):
    image = data['image']
    mask = data['segmentation_mask']

    # Redimensionar imagen y máscara
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    mask = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE))

    # Normalizar la imagen
    image = tf.cast(image, tf.float32) / 255.0

    # Convertir la máscara a un formato compatible con la red neuronal
    mask = tf.cast(mask, tf.int32)
    mask = tf.one_hot(mask, depth=3)

    # Usar solo la clase de objeto frente al fondo
    mask = mask[..., 1]  # Ignorar la clase de fondo en este caso

    return image, mask

# Aplicar el preprocesamiento a los conjuntos de datos
train_dataset = train_dataset.map(process_data)
test_dataset = test_dataset.map(process_data)

# Convertir a listas para uso posterior
X_train, y_train = [], []
X_test, y_test = [], []

for image, mask in train_dataset:
    X_train.append(image.numpy())
    y_train.append(mask.numpy())

for image, mask in test_dataset:
    X_test.append(image.numpy())
    y_test.append(mask.numpy())

# Convertir las listas a arrays numpy
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)