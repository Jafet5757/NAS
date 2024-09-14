from nats_bench import create
from colorama import Fore, Back, Style, init
import xautodl
from xautodl.models import get_cell_based_tiny_net
import matplotlib.pyplot as plt
import prettytable as pt
import math
import torch
import os

init()
os.environ['HOME'] = 'C:\\Users\\Jafet'
VERBOSE = False


class SynFlow():
    def __init__(self):
        self.scores = {}
    
    def score(self, model, input_shape, device):
      
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
        
        signs = linearize(model)

        input_dim = input_shape
        input = torch.ones((1,) + input_dim).to(device)#, dtype=torch.float64).to(device)
        output = model(input)
        torch.sum(output[0]).backward()
        
        """ for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.grad * p).detach().abs_()
            p.grad.data.zero_() """
        
        # Iteramos los pesos y los miltiplicamos por su gradiente
        for name, param in model.named_parameters():
          if param.grad is not None:
            self.scores[name] = torch.clone(param.grad * param).detach().abs_()
            param.grad.data.zero_()

        nonlinearize(model, signs)

# Create the API instance for the size search space in NATS
api = create(None, 'sss', fast_mode=True, verbose=False)

# Create the API instance for the topology search space in NATS
api = create(None, 'tss', fast_mode=True, verbose=False)

architecture_str = api.arch(12)
print(architecture_str)

# Query the loss / accuracy / time for 1234-th candidate architecture on CIFAR-10
# info is a dict, where you can easily figure out the meaning by key
info = api.get_more_info(1234, 'cifar10')
print(Fore.CYAN, info)

# Query the flops, params, latency. info is a dict.
info = api.get_cost_info(12, 'cifar10')

# Simulate the training of the 1224-th candidate:
validation_accuracy, latency, time_cost, current_total_time_cost = api.simulate_train_eval(1224, dataset='cifar10', hp='12')

# Print the information of the 12-th candidate for CIFAR-10.
print(Fore.GREEN, "validation_accuracy: ", validation_accuracy)
print("latency: ", latency)
print("time_cost: ", time_cost)
print("current_total_time_cost: ", current_total_time_cost)


all_scores = []
all_accuracies = []

for i in range(1000):
  config = api.get_net_config(i, 'cifar10')
  network = get_cell_based_tiny_net(config)
  if VERBOSE:
    print(Fore.YELLOW, config)

  # Obtenemos el accuracy, flops, parametros y latencia
  info = api.get_more_info(i, 'cifar10')
  if VERBOSE:
    print(Fore.CYAN, info, end='\n')

  if VERBOSE:
    [print(Fore.GREEN, f"{key}: {value}") for key, value in info.items()]

  # Compute synflow
  sf = SynFlow()

  # Compute the synflow score
  sf.score(network, (3, 32, 32), 'cpu')

  # Print the synflow scores
  table = pt.PrettyTable()
  table.field_names = ["Layer", "SynFlow Score"]
  for name, score in sf.scores.items():
      table.add_row([name, score.sum().item()])
  if VERBOSE:
    print(Fore.WHITE, table)

  # Print the synflow scores
  total_score = sum([score.sum().item() for score in sf.scores.values()])
  if VERBOSE:
    print(Fore.MAGENTA, f"Total SynFlow Score: {total_score}")

  all_scores.append(math.log(total_score) if total_score > 0 else 0)
  all_accuracies.append(info['test-accuracy'])

# Print resume of the synflow scores
table = pt.PrettyTable()
table.field_names = ["Model", "SynFlow Score", "Accuracy"]
for i, score in enumerate(all_scores):
    table.add_row([f"Model {i}", score, all_accuracies[i]])
print(Fore.RED, table)

# Print the average synflow score
average_score = sum(all_scores) / len(all_scores)
print(Fore.BLUE, f"Average SynFlow Score: {average_score}")

# Ordenar los scores y las accuracies
all_scores, all_accuracies = zip(*sorted(zip(all_scores, all_accuracies)))

# calculamos la correlación de Pearson
mean_score = sum(all_scores) / len(all_scores)
mean_accuracy = sum(all_accuracies) / len(all_accuracies)
cov = sum([(x - mean_score) * (y - mean_accuracy) for x, y in zip(all_scores, all_accuracies)])
std_score = math.sqrt(sum([(x - mean_score) ** 2 for x in all_scores]))
std_accuracy = math.sqrt(sum([(y - mean_accuracy) ** 2 for y in all_accuracies]))
pearson = cov / (std_score * std_accuracy)
print(Fore.GREEN, f"Pearson Correlation: {pearson}")

# Plot the synflow scores vs the accuracy
plt.scatter(all_scores, all_accuracies)
plt.xlabel("SynFlow Score")
plt.ylabel("Accuracy")
plt.title("SynFlow Score vs Accuracy")
plt.show()