import numpy as np
import tensorflow as tf
from CNN import ConvolutionalNeuralNetwork as CNN
import random
import tensorflow_datasets as tfds
import os
from dotenv import load_dotenv
import prettytable as pt
import time

# Cargamos las variables de entorno
load_dotenv(dotenv_path='./../variables.env')

seed = 6153
print('Seed:', seed)
np.random.seed(int(seed))
os.environ['PYTHONHASHSEED'] = str(seed)
tf.random.set_seed(seed)
random.seed(seed)

# Configuramos el uso de la GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


class GeneticAlgorithm:
  def __init__(self, population_size:int, max_layers:int, max_filters:int, max_kernel_size:int, generations:int, mutation_rate:float, classes:int, epochs:int, batch_size:int):
    """ 
    Genetic algorithm to optimize the architecture of a neural network
    population_size: number of individuals in the population
    max_layers: maximum number of layers
    max_filters: maximum number of filters in a layer
    max_kernel_size: maximum size of the kernel
    generations: number of generations
    mutation_rate: mutation rate 
    classes: number of classes
    """
    self.population_size = population_size
    self.max_layers = max_layers
    self.max_filters = max_filters
    self.max_kernel_size = max_kernel_size
    self.generations = generations
    self.mutation_rate = mutation_rate
    self.population = []
    self.best_individuals = []
    self.classes = classes
    self.epochs = epochs
    self.batch_size = batch_size

  def create_population(self):
    """
    Create the initial population
    """
    for _ in range(self.population_size):
      cnn_instance = CNN(epochs=self.epochs, batch_size=self.batch_size)
      cnn_instance.random_model(self.max_layers, self.max_filters, self.max_kernel_size)
      self.population.append(cnn_instance)

  def fitness(self, X_train, y_train, X_test, y_test, local_population = None):
    """
    Calculate the fitness of the population
    X_train: training dataset
    y_train: training labels
    X_test: test dataset
    y_test: test labels
    """
    for cnn_instance in (self.population if local_population is None else local_population):
      loss, accuracy = cnn_instance.compile(X_train, y_train, X_test, y_test, self.classes)
      cnn_instance.accuracy = accuracy
      cnn_instance.loss = loss

  def selection_by_tournament(self, n:int, individuals_in_tournament:int=3):
    """
    Select the best individuals by tournament
    n: number of individuals to select
    """
    # n would be pair
    n = n if n % 2 == 0 else n + 1
    # Select n individuals by tournament
    selected_one = []
    selected_two = []
    for _ in range(n):
      individuals_one = random.sample(self.population, individuals_in_tournament)
      individuals_two = random.sample(self.population, individuals_in_tournament)
      selected_one.append(max(individuals_one, key=lambda x: x.accuracy))
      selected_two.append(max(individuals_two, key=lambda x: x.accuracy))
    
    return selected_one, selected_two
  
  def crossover_by_n_pints(self, parent_one, parent_two):
    """
    Crossover by n points, just two parents
    parent_one: first parent
    parent_two: second parent
    """
    min_size = min(len(parent_one.network), len(parent_two.network))
    # Generate the number of crossover points
    n = random.randint(1, min_size - 1) if min_size > 1 else 1
    # Generate the crossover points
    crossover_points = random.sample(range(1, min_size), n) if min_size > 1 else [0]
    crossover_points.sort()
    # Initialize the children
    child_one = CNN(epochs=5, batch_size=64)
    child_two = CNN(epochs=5, batch_size=64)
    # Crossover the parents
    parent = parent_one
    for i in range(min_size):
      if i in crossover_points:
        parent = parent_two if parent == parent_one else parent_one
      child_one.network.append(parent.network[i])
    parent = parent_two
    for i in range(min_size):
      if i in crossover_points:
        parent = parent_two if parent == parent_one else parent_one
      child_two.network.append(parent.network[i])
    return child_one, child_two
  
  def crossover(self, parents_one:list, parents_two:list):
    """
    Crossover all the parents
    parent_one: first parent
    parent_two: second parent
    """
    children = []
    for i in range(len(parents_one)):
      child_one, child_two = self.crossover_by_n_pints(parents_one[i], parents_two[i])
      children.append(child_one)
      children.append(child_two)
    return children
  
  def mutation(self):
    """ 
    Muatate all the population, can be add or delete a layer, change the activation function, kernel_size or the number of filters 
    """

    # iterate over the population
    for cnn_instance in self.population:
      # iterate over the network
      for layer in cnn_instance.network:
        # mutate the activation function
        if random.random() < self.mutation_rate:
          layer['activation'] = random.choice(cnn_instance.activations)
        # mutate the number of filters
        if random.random() < self.mutation_rate:
          layer['filters'] = np.random.randint(1, self.max_filters)
        # mutate the kernel size
        if random.random() < self.mutation_rate:
          kernel_size_number = np.random.randint(1, self.max_kernel_size)
          layer['kernel_size'] = (kernel_size_number, kernel_size_number)
        # Add or delete a layer, this is a little more complex, for the moment no implement this
        """ if random.random() < self.mutation_rate:
          option = random.randint(0, 1)
          if option == 0:
            cnn_instance.add_layer(filters=np.random.randint(1, self.max_filters), kernel_size=(np.random.randint(1, self.max_kernel_size), np.random.randint(1, self.max_kernel_size)), activation=random.choice(cnn_instance.activations))
          else:
            cnn_instance.network.pop(np.random.randint(0, len(cnn_instance.network))) """
        
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

  def run(self, X_train, y_train, X_test, y_test):
    """
    Run the genetic algorithm
    X_train: training dataset
    y_train: training labels
    X_test: test dataset
    y_test: test labels
    """
    # Create the initial population
    self.create_population()
    # Calculate the fitness of the population
    self.fitness(X_train, y_train, X_test, y_test)
    # Sort the population
    self.population.sort(key=lambda x: x.accuracy, reverse=True)
    # Save the best individuals
    self.best_individuals.append(self.population[0])
    # Iterate over the generations
    for _ in range(self.generations):
      # Select the best individuals
      selected_one, selected_two = self.selection_by_tournament(self.population_size)
      # Crossover the selected individuals
      children = self.crossover(selected_one, selected_two)
      # Merge the population with the new individuals
      self.population += children
      # Mutate the population
      self.mutation()
      # Calculate the fitness of the population
      self.fitness(X_train, y_train, X_test, y_test)
      # Sort the population
      self.population.sort(key=lambda x: x.accuracy, reverse=True)
      # Save the best individuals
      self.best_individuals.append(self.population[0])
      # Select the best individuals
      self.population = self.population[:self.population_size]
      # Show the best individual
      print('Accuracy:', self.population[0].accuracy, 'Loss:', self.population[0].loss)
    # Show the best individuals
    self.show_best_individuals()


if __name__=="__main__":
  from preprocessing_dataset import X_train, y_train, X_test, y_test

  start_time = time.time()  # Get the start time

  # Create the genetic algorithm
  ga = GeneticAlgorithm(population_size=4, max_layers=5, max_filters=256, max_kernel_size=5, generations=2, mutation_rate=0.1, classes=2, epochs=5, batch_size=32)

  # Run the genetic algorithm
  ga.run(X_train, y_train, X_test, y_test)

  print("--- %s seconds ---" % (time.time() - start_time))  # Show the time spent in the process
