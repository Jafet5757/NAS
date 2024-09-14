#%%
import torch
import numpy as np
from nats_bench import create
from xautodl.models import get_cell_based_tiny_net
import torch
import matplotlib.pyplot as plt
import os
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
import time

os.environ['HOME'] = 'C:\\Users\\Jafet'
api = create(None, 'tss', fast_mode=True, verbose=False)

def gradient_scorer(network, device='cuda'):
    network.to(device)
    # Create synthetic data
    input_tensor = torch.ones((1, 3, 32, 32)).float().to(device)
    label_tensor = torch.tensor(1).unsqueeze(0).to(device)
    # Forward pass
    output = network(input_tensor)
    # Compute the loss
    loss = nn.CrossEntropyLoss()(output[1], label_tensor).to(device)
    torch.sum(loss).backward()
    
    # Compute the score
    tape=[]
    for _, param in network.named_parameters():
        tensor = torch.clone(param.grad).detach().abs_()
        tape.append(tensor.flatten())
        param.grad.data.zero_() 
    tape = torch.cat(tape)
    
    return np.log(torch.sum(tape).item()/torch.norm(tape).item())


def gradient_scorer_2(network, device='cuda'):
    network.to(device)
    # Create synthetic data
    input_tensor = torch.ones((1, 3, 32, 32)).float().to(device)
    # Forward pass
    output = network(input_tensor)
    # Compute the loss
    loss = torch.sum(output[1])
    torch.sum(loss).backward()
    
    # Compute the score
    tape=[]
    for _, param in network.named_parameters():
        tensor = torch.clone(param.grad * param).detach().abs_()
        tape.append(tensor.flatten())
        param.grad.data.zero_() 
    tape = torch.cat(tape)
    
    return np.log(torch.sum(tape).item())

#%%
accs = []
scores = []
timer = time.time()

for i in range(1000):
    config = api.get_net_config(i, 'cifar10')
    network = get_cell_based_tiny_net(config)
    
    accuracy = api.get_more_info(i, 'cifar10')['test-accuracy']
    accs.append(accuracy)
    
    score = gradient_scorer(network)
    scores.append(score)

plt.scatter(scores, accs)
plt.xlabel('Scores')
plt.ylabel('Accuracies')
plt.title('Cifar10')
plt.show()

pearson, _ = pearsonr(scores, accs)
spearman, _ = spearmanr(scores, accs)
print(f'Pearson correlation: {pearson} - Spearman correlation: {spearman}')
print(f'Time elapsed: {time.time()-timer}')
# %%
accs = []
scores = []
timer = time.time()

for i in range(1000):
    config = api.get_net_config(i, 'cifar100')
    network = get_cell_based_tiny_net(config)
    
    accuracy = api.get_more_info(i, 'cifar100')['test-accuracy']
    accs.append(accuracy)
    
    score = gradient_scorer(network)
    scores.append(score)

plt.scatter(scores, accs)
plt.xlabel('Scores')
plt.ylabel('Accuracies')
plt.title('Cifar100')
plt.show()

pearson, _ = pearsonr(scores, accs)
spearman, _ = spearmanr(scores, accs)
print(f'Pearson correlation: {pearson} - Spearman correlation: {spearman}')
print(f'Time elapsed: {time.time()-timer}')
# %%
accs = []
scores = []
timer = time.time()

for i in range(1000):
    config = api.get_net_config(i, 'ImageNet16-120')
    network = get_cell_based_tiny_net(config)
    
    accuracy = api.get_more_info(i, 'ImageNet16-120')['test-accuracy']
    accs.append(accuracy)
    
    score = gradient_scorer(network)
    scores.append(score)

plt.scatter(scores, accs)
plt.xlabel('Scores')
plt.ylabel('Accuracies')
plt.title('imageNet16-120')
plt.show()

pearson, _ = pearsonr(scores, accs)
spearman, _ = spearmanr(scores, accs)
print(f'Pearson correlation: {pearson} - Spearman correlation: {spearman}')
print(f'Time elapsed: {time.time()-timer}')