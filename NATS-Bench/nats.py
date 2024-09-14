#%%
import torch
import random
import numpy as np
from nats_bench import create
from xautodl.models import get_cell_based_tiny_net
import torch
#from Scorers import get_input_tapes, get_weight_tapes, cov_estimator
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import torch.nn as nn
import math
import time
#%%
os.environ['HOME'] = 'C:\\Users\\Jafet'
api = create(None, 'tss', fast_mode=True, verbose=False)
loss_function = nn.CrossEntropyLoss()
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
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
    original_values = linearize(model)
    
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
    nonlinearize(model, original_values)
    
    total_score = sum([score.sum().item() for score in scores.values()])
    
    return total_score


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


def get_weight_tapes(network, inputs, labels):
    tapes = []
    for input_, label in zip(inputs, labels):
        # Preprocess the input and output
        input_tensor = np.transpose(input_, (2, 0, 1))
        input_tensor = torch.tensor(input_tensor).unsqueeze(0).float()
        label_tensor = torch.tensor(label)
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

def score_nets(n_networks, batch_size, estimator, tapes_function):
    accs = []
    scores = []
    
    #Random batch
    indices = random.sample(range(len(X_train)), batch_size)
    X_train_batch = X_train[indices]
    y_train_batch = y_train[indices]
    
    index = random.randint(0, 10000)
    for i in range(index, index + n_networks):
        config = api.get_net_config(i, 'cifar10')
        network = get_cell_based_tiny_net(config)
        
        accuracy = api.get_more_info(i, 'cifar10')['test-accuracy']
        accs.append(accuracy)
        
        tapes = tapes_function(network, X_train_batch, y_train_batch)
        score = estimator(tapes)
        scores.append(score)
    
    return accs, scores



def score_nets_synflow(n_networks, synflow_scorer):
    accs = []
    scores = []
    
    
    index = random.randint(0, 10000)
    for i in range(index, index + n_networks):
        config = api.get_net_config(i, 'cifar10')
        network = get_cell_based_tiny_net(config)
        
        accuracy = api.get_more_info(i, 'cifar10')['test-accuracy']
        accs.append(accuracy)
        
        score = synflow_scorer(network)
        scores.append(score)
    
    return accs, scores

def get_pearson_correlation(scores, accs):
    mean_score = sum(scores) / len(scores)
    mean_accuracy = sum(accs) / len(accs)
    cov = sum([(x - mean_score) * (y - mean_accuracy) for x, y in zip(scores, accs)])
    std_score = math.sqrt(sum([(x - mean_score) ** 2 for x in scores]))
    pearson_correlation = cov / (std_score * math.sqrt(sum([(y - mean_accuracy) ** 2 for y in accs])))
    return pearson_correlation

#%% TESTING
# Tomamos el tiempo
start = time.time()
accs, scores = score_nets(1500, 20, cosine_estimator, get_weight_tapes)

plt.scatter(scores, accs)
plt.xlabel('Scores')
plt.ylabel('Accuracies')
plt.title('Gradient similarity stimator')
plt.show()
print(f'Pearson correlation: {get_pearson_correlation(scores, accs)}')
print(f'Time: {time.time() - start}')

# %%
accs, scores = score_nets_synflow(1500, synflow_scorer_abs)
# tomamos el tiempo
start = time.time()

plt.scatter(scores, accs)
plt.xlabel('Scores')
plt.ylabel('Accuracies')
plt.title('Synflow abs estimator')
plt.show()
print(f'Pearson correlation: {get_pearson_correlation(scores, accs)}')
print(f'Time: {time.time() - start}')

# %%
accs, scores = score_nets_synflow(1500, synflow_scorer_ones)
# tomamos el tiempo
start = time.time()

plt.scatter(scores, accs)
plt.xlabel('Scores')
plt.ylabel('Accuracies')
plt.title('Synflow ones estimator')
plt.show()
print(f'Pearson correlation: {get_pearson_correlation(scores, accs)}')
print(f'Time: {time.time() - start}')

# %%
