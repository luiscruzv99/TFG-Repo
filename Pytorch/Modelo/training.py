import torch
import torch.nn as nn


def train(device, dataset, net, optimizer,
          loss_fn=nn.MSELoss(reduction='mean')):
    '''
    Funcion que entrena una red neuronal, sobre un dataset, con un optimizador
    y funcion de perdida pasados como parametros. Se encarga de hacer la
    propagacion hacia adelante y hacia atra
    '''

    # Recorremos todos los ejemplos con sus etiquetas
    # y entrenamos la red con ellos
    for example, label in dataset:
        optimizer.zero_grad()  # Puesta a 0 de los gradientes de la red
        output = net.forward(example.to(device))  # Propagacion hacia adelante
        loss = loss_fn(output, label.to(device))  # Calculo de la perdida
        loss.backward()  # Propagacion hacia atras
        optimizer.step()  # Optimizacion de la pasada hacia atras


def validate_or_test(device, dataset, net, optimizer,
                     loss_fn=nn.MSELoss(reduction='mean')):
    '''
    Funcion que valida o testea la red neuronal pasada como parametro (tambien
    se deben pasar la funcion de perdida y el optimizador como parametros)
    sobre un dataset pasado com parametro. Esta funcion calcula además la
    precision y la perdida memdia de la red.
    '''

    # Ponemos la red en modo evaluacion
    net.eval()

    acc = 0.0  # Precision de la red
    avg_loss = 0.0  # Perdida media de la red

    # Recorremos todos los ejemplos con sus etiquetas
    for example, label in dataset:
        optimizer.zero_grad()  # Puesta a 0 de los gradientes de la red
        output = net.forward(example.to(device))  # Propagacion hacia adelante
        loss = loss_fn(output, label.to(device))  # Calculo de la perdida

        guess = torch.max(output, -1)[1]  # Solucion que elige la red
        true_val = torch.max(label, -1)[1]  # Solucion correcta al ejemplo

        # Si coinciden, anhadir uno a la precision
        if(guess.item() == true_val.item()):
            acc += 1

        avg_loss += loss.item()  # Acumulacion de la perdida

    # Calculamos la precision media y la perdida media y las devolvemos
    acc = acc / len(dataset)
    avg_loss = avg_loss / len(dataset)

    return [acc, avg_loss]
