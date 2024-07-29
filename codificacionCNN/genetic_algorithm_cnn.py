import numpy as np
import tensorflow as tf
from CNN import ConvolutionalNeuralNetwork as CNN
import random
import tensorflow_datasets as tfds
import os
from dotenv import load_dotenv

# Cargamos las variables de entorno
load_dotenv(dotenv_path='./../variables.env')

# set the seed
seed = os.getenv('SEED')
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
tf.random.set_seed(seed)


class GeneticAlgorithm:
  def __init__(self, population_size:int, max_layers:int, max_filters:int, generations:int, mutation_rate:float):
    pass