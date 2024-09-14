import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import prettytable as pt

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
        torch.sum(output).backward()
        
        """ for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.grad * p).detach().abs_()
            p.grad.data.zero_() """
        
        # Iteramos los pesos y los miltiplicamos por su gradiente
        for name, param in model.named_parameters():
            self.scores[name] = torch.clone(param.grad * param).detach().abs_()
            param.grad.data.zero_()

        nonlinearize(model, signs)


# Definir la arquitectura de la red neuronal
class CustomDenseNN(nn.Module):
    def __init__(self):
        super(CustomDenseNN, self).__init__()
        
        # Definir la capa de aplanado (similar a Flatten en Keras)
        self.flatten = nn.Flatten()

        # Definir las capas densas (equivalente a Dense en Keras)
        self.fc1 = nn.Linear(in_features=28*28, out_features=33)
        self.fc2 = nn.Linear(in_features=33, out_features=88)
        self.fc3 = nn.Linear(in_features=88, out_features=10)
        
        # Inicializar los pesos y los sesgos como 'ones'
        nn.init.constant_(self.fc1.weight, 1)
        nn.init.constant_(self.fc1.bias, 1)
        nn.init.constant_(self.fc2.weight, 1)
        nn.init.constant_(self.fc2.bias, 1)
        nn.init.constant_(self.fc3.weight, 1)
        nn.init.constant_(self.fc3.bias, 1)
        
    def forward(self, x):
        # Aplanar la entrada
        x = self.flatten(x)
        # Pasar por las capas densas con activaciones correspondientes
        x = F.elu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

# Definir la arquitectura de la red neuronal
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        
        # Definir las capas convolucionales
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 1), padding="same")
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding="same")
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding="same")
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding="same")
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding="same")
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding="same")
        
        # Inicializar los pesos y los sesgos como 'ones'
        nn.init.constant_(self.conv1.weight, 1)
        nn.init.constant_(self.conv1.bias, 1)
        nn.init.constant_(self.conv2.weight, 1)
        nn.init.constant_(self.conv2.bias, 1)
        nn.init.constant_(self.conv3.weight, 1)
        nn.init.constant_(self.conv3.bias, 1)
        nn.init.constant_(self.conv4.weight, 1)
        nn.init.constant_(self.conv4.bias, 1)
        nn.init.constant_(self.conv5.weight, 1)
        nn.init.constant_(self.conv5.bias, 1)
        nn.init.constant_(self.conv6.weight, 1)
        nn.init.constant_(self.conv6.bias, 1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        return x
    
# Definir la arquitectura de la red neuronal
class CustomCNN2(nn.Module):
    def __init__(self):
        super(CustomCNN2, self).__init__()
        
        # Definir las capas convolucionales con padding="same"
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 1), padding="same")
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding="same")
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding="same")
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding="same")
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding="same")
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), padding="same")
        
        # Inicializar los pesos y los sesgos como 'ones'
        nn.init.constant_(self.conv1.weight, 1)
        nn.init.constant_(self.conv1.bias, 1)
        nn.init.constant_(self.conv2.weight, 1)
        nn.init.constant_(self.conv2.bias, 1)
        nn.init.constant_(self.conv3.weight, 1)
        nn.init.constant_(self.conv3.bias, 1)
        nn.init.constant_(self.conv4.weight, 1)
        nn.init.constant_(self.conv4.bias, 1)
        nn.init.constant_(self.conv5.weight, 1)
        nn.init.constant_(self.conv5.bias, 1)
        nn.init.constant_(self.conv6.weight, 1)
        nn.init.constant_(self.conv6.bias, 1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        return x

# Crear una instancia de la red
model = CustomCNN2()

# Definir un optimizador y una función de pérdida (opcional en esta fase)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Ejemplo de un ciclo de entrenamiento
""" for epoch in range(5):  # 5 épocas de ejemplo
    # Entrada de ejemplo: 64 imágenes de 28x28 aplanadas
    inputs = torch.randn(64, 784)  # Batch de 64 imágenes aleatorias
    labels = torch.randint(0, 10, (64,))  # Etiquetas aleatorias para 10 clases
    
    # Paso hacia adelante
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
    # Paso hacia atrás y optimización
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f'Epoch [{epoch+1}/5], Loss: {loss.item():.4f}') """

# Crear una instancia de SynFlow
sf = SynFlow()
sf.score(model, (1, 784), 'cpu')

# Crear una tabla para mostrar los scores por capa
table = pt.PrettyTable()
table.field_names = ['Layer', 'Score']
for name, score in sf.scores.items():
    table.add_row([name, score.sum().item()])
print(table)

# Score total
total_score = sum(score.sum().item() for score in sf.scores.values())
print(f'Total score: {total_score:.4f}')
