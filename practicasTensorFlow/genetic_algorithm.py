import numpy as np
import tensorflow as tf
import neural_evolution as nn
import random
from tensorflow.keras.datasets import mnist
import prettytable as pt
import os
from dotenv import load_dotenv

# Cargamos las variables de entorno
load_dotenv(dotenv_path='./../variables.env')

# set the seed
seed = os.getenv('SEED')
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
tf.random.set_seed(seed)

class GeneticAlgorithm:
  def __init__(self, population_size:int, max_layers:int, max_units:int, generations:int, mutation_rate:float):
    """ 
    Genetic algorithm to optimize the architecture of a neural network
    population_size: number of individuals in the population
    max_layers: maximum number of layers
    max_units: maximum number of neurons in a layer
    generations: number of generations
    mutation_rate: mutation rate 
    """
    self.population_size = population_size
    self.max_layers = max_layers
    self.max_units = max_units
    self.generations = generations
    self.mutation_rate = mutation_rate
    self.population = []
    self.best_individuals = []

  def create_population(self):
    """
    Create the initial population
    """
    for _ in range(self.population_size):
      nn_instance = nn.NeuralNetwork()
      nn_instance.random_initialization(self.max_layers, self.max_units)
      self.population.append(nn_instance)

  def fitness(self, X_train, y_train, X_test, y_test, classes, local_population = None):
    """
    Calculate the fitness of the population
    X_train: training dataset
    y_train: training labels
    X_test: test dataset
    y_test: test labels
    classes: number of classes
    """
    for nn_instance in (self.population if local_population is None else local_population):
      loss, accuracy = nn_instance.compile(X_train, y_train, X_test, y_test, classes)
      nn_instance.accuracy = accuracy
      nn_instance.loss = loss

  def selection_by_tournament(self, n:int):
    """
    Select n individuals by tournament
    n: number of individuals to select
    """
    selected_one = []
    selected_two = []
    # shuffle the population
    random.shuffle(self.population)
    # iterate and select the best two individuals
    for i in range(0, len(self.population), 2):
      # select this and the next individual
      if self.population[i].accuracy > self.population[i+1].accuracy:
        selected_one.append(self.population[i])
      else:
        selected_one.append(self.population[i+1])
    # shuffle the population
    random.shuffle(self.population)
    # iterate and select the best two individuals
    for i in range(0, len(self.population), 2):
      # select this and the next individual
      if self.population[i].accuracy > self.population[i+1].accuracy:
        selected_two.append(self.population[i])
      else:
        selected_two.append(self.population[i+1])
    return selected_one, selected_two
  
  def show_best_individuals(self):
    """
    Show the best individuals
    """
    table = pt.PrettyTable()
    table.field_names = ['Generation', 'Accuracy', 'Loss', 'Architecture']
    c = 1
    for individual in self.best_individuals:
      table.add_row([c, individual.accuracy, individual.loss, individual.network])
      c += 1
    print(table)

  def uniform_crossover(self, parents_one:list, parents_two:list):
    """
    Uniform crossover
    parent_one: first parent
    parent_two: second parent
    """
    # create the child
    childs = []
    # iterate over the parents
    for parent_one, parent_two in zip(parents_one, parents_two):\
      # create the child
      child = nn.NeuralNetwork()
      # iterate each layer
      for layer_one, layer_two in zip(parent_one.network, parent_two.network):
        # select the layer of one of the parents
        if random.random() > 0.5:
          child.network.append(layer_one)
        else:
          child.network.append(layer_two)
      childs.append(child)
    return childs
  
  def mutation(self):
    """
    Mutation of the population
    """
    # iterate over the population
    for nn_instance in self.population:
      # iterate over the layers
      for layer in nn_instance.network:
        # mutate the layer
        if random.random() < self.mutation_rate:
          layer[0] = np.random.randint(1, self.max_units)
          layer[1] = np.random.choice(nn_instance.activations)
        # add or remove a layer
        if random.random() < self.mutation_rate:
          if random.random() < 0.5:
            nn_instance.add_layer(np.random.randint(1, self.max_units), np.random.choice(nn_instance.activations))
          else:
            nn_instance.network.pop(np.random.randint(0, len(nn_instance.network)))

  def run(self, X_train, y_train, X_test, y_test, classes):
    """
    Run the genetic algorithm
    X_train: training dataset
    y_train: training labels
    X_test: test dataset
    y_test: test labels
    classes: number of classes
    """
    # create the initial population
    self.create_population()
    # iterate over the generations
    for _ in range(self.generations):
      # calculate the fitness of the population
      self.fitness(X_train, y_train, X_test, y_test, classes)
      # select the best individuals
      selected_one, selected_two = self.selection_by_tournament(2)
      # crossover the selected individuals
      childs = self.uniform_crossover(selected_one, selected_two)
      # mutate the population
      self.mutation()
      # calculate the fitness of the new population
      self.fitness(X_train, y_train, X_test, y_test, classes, childs)
      # join the population with the new individuals
      self.population += childs
      # sort the population by accuracy
      self.population = sorted(self.population, key=lambda x: x.accuracy, reverse=True)
      # select the best individuals
      self.population = self.population[:self.population_size]
      # print the best individual
      print('Accuracy:', self.population[0].accuracy, 'Loss:', self.population[0].loss)
      self.best_individuals.append(self.population[0])
    # show the best individuals
    self.show_best_individuals()

if __name__ == '__main__':
  from tensorflow.keras.utils import to_categorical

  # load the dataset
  (X_train, y_train), (X_test, y_test) = mnist.load_data()

  # Normalizar las imágenes a un rango de 0 a 1
  X_train = X_train.astype('float32') / 255.0
  X_test = X_test.astype('float32') / 255.0

  # Si las imágenes son de una sola canal (escala de grises), necesitas agregar una dimensión extra
  # para que sean compatibles con las capas de convolución de Keras
  X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
  X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))


  # create the genetic algorithm
  ga = GeneticAlgorithm(4, 4, 100, 6, 0.15)
  # run the genetic algorithm
  ga.run(X_train, y_train, X_test, y_test, 10)