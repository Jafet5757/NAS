#%%
import torch
import random
import numpy as np
from nats_bench import create
from xautodl.models import get_cell_based_tiny_net
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
import os
import tensorflow as tf
import torch.nn as nn
import math
import time
import pandas as pd
#%%
os.environ['HOME'] = 'C:\\Users\\Jafet'
api = create(None, 'tss', fast_mode=True, verbose=False)
loss_function = nn.CrossEntropyLoss()

# Para cifar10
(X_train_10, y_train_10), (X_test_10, y_test_10) = tf.keras.datasets.cifar10.load_data()

# Para cifar100
(X_train_100, y_train_100), (X_test_100, y_test_100) = tf.keras.datasets.cifar100.load_data()

#%%
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
print(dev)
device = torch.device(dev)


#%%

#synflow

def synflow_scorer_ones(model):
    @torch.no_grad()
    def linearize(model):
        original_values = {}
        for name, param in model.state_dict().items():
            original_values[name] = param.clone()  # Guarda una copia del valor original
            param.copy_(torch.ones_like(param))  # Cambia los valores de los parámetros a unos
        return original_values
    @torch.no_grad()
    def nonlinearize(model, original_values):
        for name, param in model.state_dict().items():
            param.copy_(original_values[name])

    scores = {}
    # Set ones
    #original_values = linearize(model)
    
    # Forward pass
    input = torch.ones((1, 3, 32, 32)).float()
    output = model(input)
    
    # hacemos retropropagación
    torch.sum(output[1]).backward()
    
    # Compute the scores
    for name, param in model.named_parameters():
        scores[name] = torch.clone(param.grad).detach().abs_()
        param.grad.data.zero_()
    
    # Reset the model
    #nonlinearize(model, original_values)
    
    total_score = np.log(sum([score.sum().item() for score in scores.values()]))
    
    return total_score


def gradient_scorer(network):
    # Create synthetic data
    input_tensor = torch.ones((1, 3, 32, 32)).float()
    label_tensor = torch.tensor(1).unsqueeze(0)
    # Forward pass
    output = network(input_tensor)
    # Compute the loss
    loss = nn.CrossEntropyLoss()(output[1], label_tensor)
    torch.sum(loss).backward(retain_graph=True)
    
    # Compute the score
    tape=[]
    for _, param in network.named_parameters():
        tensor = torch.clone(param.grad).detach().abs_()
        tape.append(tensor.flatten())
        param.grad.data.zero_() 
    tape = torch.cat(tape)
    
    
    score = np.log(tape.sum().item()/torch.norm(tape).item())
    return score

def synflow_scorer_abs(model):
    
    @torch.no_grad()
    def linearize(model):
        # model.double()
        signs = {}
        for name, param in model.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs
    @torch.no_grad()
    def nonlinearize(model, signs):
        # model.float()
        for name, param in model.state_dict().items():
            param.mul_(signs[name])

    scores = {}
    # Set ones
    signs = linearize(model)
    
    # Forward pass
    input = torch.ones((1, 3, 32, 32)).float()
    output = model(input)
    
    # Backward pass
    loss = torch.sum(output[1])
    loss.backward()
    
    # Compute the scores
    for name, param in model.named_parameters():
        if param.grad is not None:
            scores[name] = torch.clone(param.grad * param).detach().abs_()
            param.grad.data.zero_()
    
    # Reset the model
    nonlinearize(model, signs)
    
    total_score = sum([score.sum().item() for score in scores.values()])
    
    return total_score


def get_weight_tapes(network, inputs, labels, device="cuda"):
  tapes = []
  network.to(device)  # Mueve la red a la GPU

  # Mueve los inputs y labels a la GPU
  inputs = [torch.tensor(np.transpose(input_, (2, 0, 1))).unsqueeze(0).float().to(device) for input_ in inputs]
  labels = [torch.tensor(label.astype(np.uint8)).to(device) for label in labels]

  for input_tensor, label_tensor in zip(inputs, labels):
      # Forward pass
      output = network(input_tensor)
      
      # Compute the loss
      loss = loss_function(output[1], label_tensor)
      
      # Backward pass
      torch.sum(loss).backward()
      
      # Save the gradients
      tape = []
      for name, param in network.named_parameters():
          if param.grad is not None:
              tensor = torch.clone(param.grad).detach().abs_()
              tape.append(tensor.flatten())
              param.grad.data.zero_()
      tape = torch.cat(tape)
      tapes.append(tape)
    
  return tapes

def get_input_tapes(network, inputs, __):
    tapes = []
    
    for input_ in inputs:
        input_tensor = np.transpose(input_, (2, 0, 1))
        input_tensor = torch.tensor(input_tensor).unsqueeze(0).float()
        input_tensor.requires_grad = True
        output = network(input_tensor)
        output = torch.sum(output[1])
        output.backward()
        tensor = input_tensor.grad
        tapes.append(tensor.flatten())
    
    return tapes



def calculate_hamming_distance(code1, code2):
    """Calcula la distancia de Hamming entre dos códigos binarios."""
    return np.sum(code1 != code2)

def calculate_kernel_matrix(activations):
    """
    Calcula la matriz K_H utilizando las activaciones binarizadas.
    activations: matriz (N, num_neurons), donde cada fila es el código binario para una entrada.
    """
    N = activations.shape[0]
    KH = np.zeros((N, N))
    
    # Calcular la distancia de Hamming entre cada par de códigos binarios
    for i in range(N):
        for j in range(N):
            KH[i, j] = calculate_hamming_distance(activations[i], activations[j])
    
    # Normalizar para que los elementos diagonales sean 1 (como en el artículo)
    KH = N - KH
    KH /= KH.max()
    
    return KH

def calculate_jacobian_score(model, device='cuda', dataset='cifar10', batch_size=10):
    """
    Calcula el score del modelo usando la técnica de NASWOT descrita en el artículo.
    
    model: Red neuronal no entrenada.
    data_loader: Dataloader con las entradas (inputs) para calcular las activaciones.
    device: Dispositivo en el que se ejecuta el cálculo (e.g., 'cuda' o 'cpu').
    
    Retorna:
        score: El score del modelo basado en las activaciones iniciales.
    """
    model.to(device)
    model.eval()

    all_activations = []

    if dataset == 'cifar10':
        X_train = X_train_10
        y_train = y_train_10
    elif dataset == 'cifar100':
        X_train = X_train_100
        y_train = y_train_100

    #Random batch
    indices = random.sample(range(len(X_train)), batch_size)
    X_train_batch = X_train[indices]
    y_train_batch = y_train[indices]

    with torch.no_grad():
        for inputs in X_train_batch:
            # Enviamos la entrada a la GPU
            inputs = torch.tensor(np.transpose(inputs, (2, 0, 1))).unsqueeze(0).float().to(device)
            
            # Pasar los datos por el modelo
            y, outputs = model(inputs)

            # Obtener activaciones de las capas ReLU
            activations = (outputs > 0).cpu().numpy().astype(int)  # Binarizar las activaciones (ReLU > 0)
            all_activations.append(activations)
    
    # Concatenar todas las activaciones
    all_activations = np.concatenate(all_activations, axis=0)
    
    # Calcular la matriz de Hamming K_H
    KH = calculate_kernel_matrix(all_activations)
    
    # Calcular el determinante de la matriz K_H
    score = np.log(np.linalg.det(KH) + 1e-10)  # Se añade una pequeña constante para evitar log(0)

    # si es nan, se retorna 0
    if np.isnan(score):
        return 0

    return score


def get_covariance(x, y):
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    n = len(x)
    covariance = torch.sum((x - x_mean) * (y - y_mean)) / n
    return covariance

def cov_estimator(tapes):
    cov_estimation = 0
    for i in range(len(tapes)):
        for j in range(len(tapes)):
            cov_estimation += get_covariance(tapes[i], tapes[j])
    cov_estimation = cov_estimation/(len(tapes)**2)
    return float(cov_estimation)

def get_euclidean_distance(x, y):
    return torch.norm(x - y)

def euclidean_estimator(tapes):
    euclidean_estimation = 0
    for i in range(len(tapes)):
        for j in range(len(tapes)):
            euclidean_estimation += get_euclidean_distance(tapes[i], tapes[j])
    euclidean_estimation = euclidean_estimation/(len(tapes)**2)
    return float(euclidean_estimation)

def get_cosine_similarity(x, y):
    return torch.dot(x, y) / (torch.norm(x) * torch.norm(y))

def cosine_estimator(tapes):
    cosine_estimation = 0
    for i in range(len(tapes)):
        for j in range(len(tapes)):
            cosine_estimation += get_cosine_similarity(tapes[i], tapes[j])
    cosine_estimation = cosine_estimation/(len(tapes)**2)
    return float(cosine_estimation)

def score_nets(n_networks, batch_size, estimator, tapes_function, dataset='cifar10', index=range(100)):
    accs = []
    scores = []
    if dataset == 'cifar10':
        X_train = X_train_10
        y_train = y_train_10
    elif dataset == 'cifar100':
        X_train = X_train_100
        y_train = y_train_100
        
    #Random batch
    indices = random.sample(range(len(X_train)), batch_size)
    X_train_batch = X_train[indices]
    y_train_batch = y_train[indices]
    
    for i in index:
        config = api.get_net_config(i, dataset)
        network = get_cell_based_tiny_net(config)
        
        accuracy = api.get_more_info(i, dataset)['test-accuracy']
        accs.append(accuracy)
        
        tapes = tapes_function(network, X_train_batch, y_train_batch)
        score = estimator(tapes)
        scores.append(score)
    
    return accs, scores



def score_nets_synflow(n_networks, synflow_scorer, dataset='cifar10', index=range(100)):
    accs = []
    scores = []
    
    for i in index:
        config = api.get_net_config(i, dataset)
        network = get_cell_based_tiny_net(config)
        
        accuracy = api.get_more_info(i, dataset)['test-accuracy']
        accs.append(accuracy)
        
        score = synflow_scorer(network)
        scores.append(score)
    
    return accs, scores

def score_nets_jacobian(n_networks, dataset='cifar10'):
    accs = []
    scores = []
    
    index = random.sample(range(15000), n_networks)
    for i in index:
        config = api.get_net_config(i, dataset)
        network = get_cell_based_tiny_net(config)
        
        accuracy = api.get_more_info(i, dataset)['test-accuracy']
        accs.append(accuracy)
        
        score = calculate_jacobian_score(network)
        scores.append(score)
    
    return accs, scores

def score_nets_gradient_scorer(n_networks, dataset='cifar10', index=range(100)):
    accs = []
    scores = []
    
    for i in index:
        config = api.get_net_config(i, dataset)
        network = get_cell_based_tiny_net(config)
        
        accuracy = api.get_more_info(i, dataset)['test-accuracy']
        accs.append(accuracy)
        
        score = gradient_scorer(network)
        scores.append(score)
    
    return accs, scores

def get_pearson_correlation(scores, accs):
    return pearsonr(scores, accs)[0]

def get_spearman_correlation(scores, accs):
    return spearmanr(scores, accs)[0]

def get_dispertion(scores, accs):
    mean_score = sum(scores) / len(scores)
    mean_accuracy = sum(accs) / len(accs)
    dispertion = math.sqrt(sum([(x - mean_score) ** 2 for x in scores]) + sum([(y - mean_accuracy) ** 2 for y in accs]))
    return dispertion

#%%
def write_acc_and_scores(accs, scores, name="results"):
    df = pd.DataFrame({'Accuracy': accs, 'Score': scores})
    df.to_csv(name + '.csv', index=True)

# Cantidad de redes
gradiente = 10000
synflow = 10000
grd_scorer = 10000
general = 15000

# Usamos los mismo índices para las tres pruebas
index = random.sample(range(15000), general)

#%% TESTING
# start time
start = time.time()
accs, scores = score_nets(gradiente, 2, cosine_estimator, get_weight_tapes, 'cifar10', index)

# Escribir los resultados
write_acc_and_scores(accs, scores, 'results_gradiente_cosine_estimator_cifar10')

plt.scatter(scores, accs)
plt.xlabel('Scores')
plt.ylabel('Accuracies')
plt.title('Gradient similarity stimator CIFAR10')
plt.show()
print(f'Pearson correlation: {get_pearson_correlation(scores, accs)}')
print(f'Spearman correlation: {get_spearman_correlation(scores, accs)}')
print(f'Time: {time.time() - start}')

# %%
# start time
start = time.time()
accs, scores = score_nets_gradient_scorer(synflow, 'cifar10', index)

# Escribir los resultados
write_acc_and_scores(accs, scores, 'results_gradient_scorer_cifar10')

plt.scatter(scores, accs)
plt.xlabel('Scores')
plt.ylabel('Accuracies')
plt.title('Gradient scorer CIFAR10')
plt.show()
print(f'Pearson correlation: {get_pearson_correlation(scores, accs)}')
print(f'Spearman correlation: {get_spearman_correlation(scores, accs)}')
print(f'Time: {time.time() - start}')

# %%
# start time
start = time.time()
accs, scores = score_nets_synflow(synflow, synflow_scorer_ones, 'cifar10', index)

# Escribir los resultados
write_acc_and_scores(accs, scores, 'results_synflow_ones_estimator_cifar10')

plt.scatter(scores, accs)
plt.xlabel('Scores')
plt.ylabel('Accuracies')
plt.title('Synflow ones estimator CIFAR10')
plt.show()
print(f'Pearson correlation: {get_pearson_correlation(scores, accs)}')
print(f'Spearman correlation: {get_spearman_correlation(scores, accs)}')
print(f'Time: {time.time() - start}')

# %%
# start time
start = time.time()
accs, scores = score_nets(gradiente, 2, cosine_estimator, get_weight_tapes, 'cifar100', index)

# Escribir los resultados
write_acc_and_scores(accs, scores, 'results_gradiente_cosine_estimator_cifar100')

plt.scatter(scores, accs)
plt.xlabel('Scores')
plt.ylabel('Accuracies')
plt.title('Gradient similarity stimator CIFAR100')
plt.show()
print(f'Pearson correlation: {get_pearson_correlation(scores, accs)}')
print(f'Spearman correlation: {get_spearman_correlation(scores, accs)}')
print(f'Time: {time.time() - start}')

# %%
# start time
start = time.time()
accs, scores = score_nets_gradient_scorer(synflow, 'cifar100', index)

# Escribir los resultados
write_acc_and_scores(accs, scores, 'results_gradient_scorer_cifar100')

plt.scatter(scores, accs)
plt.xlabel('Scores')
plt.ylabel('Accuracies')
plt.title('Gradient scorer CIFAR100')
plt.show()
print(f'Pearson correlation: {get_pearson_correlation(scores, accs)}')
print(f'Spearman correlation: {get_spearman_correlation(scores, accs)}')
print(f'Time: {time.time() - start}')

# %%
# start time
start = time.time()
accs, scores = score_nets_synflow(synflow, synflow_scorer_ones, 'cifar100', index)

# Escribir los resultados
write_acc_and_scores(accs, scores, 'results_synflow_ones_estimator_cifar100')

plt.scatter(scores, accs)
plt.xlabel('Scores')
plt.ylabel('Accuracies')
plt.title('Synflow ones estimator CIFAR100')
plt.show()
print(f'Pearson correlation: {get_pearson_correlation(scores, accs)}')
print(f'Spearman correlation: {get_spearman_correlation(scores, accs)}')
print(f'Time: {time.time() - start}')

# %%
# Ordenamos los acc de menor a mayor y los mostramos
accs.sort()

plt.plot(range(len(accs)), accs)
plt.xlabel('Networks')
plt.ylabel('Accuracies')
plt.title('Accuracies sorted CIFAR100')
plt.show()