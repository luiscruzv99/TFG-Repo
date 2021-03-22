import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import Modelo.model  # Implementacion de ResNet34 definida en el archivo model.py


'''
Funcion que entrena una red neuronal, sobre un dataset, con un optimizador y funcion de perdida
pasados como parametros. Se encarga de hacer la propagacion hacia adelante y hacia atras
'''
def train(device, dataset, net, optimizer, loss_fn=nn.MSELoss(reduction='mean')):

    # Recorremos todos los ejemplos con sus etiquetas y entrenamos la red con ellos
    for example, label in dataset:
        optimizer.zero_grad()
        output = net.forward(example.to(device))
        loss = loss_fn(output, label.to(device))
        loss.backward()
        optimizer.step()


'''
Funcion que valida o testea la red neuronal pasada como parametro (tambien se deben pasar la funcion de perdida 
y el optimizador como parametros) sobre un dataset pasado com parametro.
Esta funcion calcula además la precision y la pérdida memdia de la red.
'''
def validate_or_test(device, dataset, net, optimizer, loss_fn=nn.MSELoss(reduction='mean')):
    
    # Ponemos la red en modo evaluacion
    net.eval()

    acc = 0.0  # Precision de la red
    avg_loss = 0.0  # Perdida media de la red

    # Recorremos todos los ejemplos con sus etiquetas
    for example, label in dataset:
        optimizer.zero_grad()
        output = net.forward(example.to(device))
        loss = loss_fn(output, label.to(device))

        guess = torch.max(output, -1)[1]
        true_val = torch.max(label, -1)[1]

        if(guess.item() == true_val.item()):
            acc += 1
        avg_loss += loss.item()
    
    # Calculamos la precision y la perdida y las devolvemos
    acc = acc / len(dataset)
    avg_loss = avg_loss / len(dataset)

    return [acc, avg_loss]
