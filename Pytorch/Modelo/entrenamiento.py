import torch
import torch.nn as nn


def entrena(dispositivo, dataset, red, optimizador,
          perdida_fn=nn.MSELoss(reduction='mean')):
    """
    Funcion que entrena una red neuronal, sobre un dataset, con un optimizador
    y funcion de perdida pasados como parametros. Se encarga de hacer la
    propagacion hacia adelante y hacia atras

    @type dispositivo: str
    @param dispositivo: el dispositivo al que mandar los datos, en el que se
           encuentra el modelo

    @type dataset: DataLoader
    @param dataset: lista con las muestras y las etiquetas a enseñar a la red,
           distribuidas en batches

    @type red: ResNet
    @param red: modelo de la red neuronal ResNet

    @type optimizador: Optimizer
    @param optimizador: el tipo de algoritmo de descenso gradiente a utilizar

    @type perdida_fn: Loss
    @param perdida_fn: funcion de perdida a utilizar para ensenhar a la red
    """

    # Recorremos todos los ejemplos con sus etiquetas
    # y entrenamos la red con ellos
    for example, label in dataset:
        optimizador.zero_grad()  # Puesta a 0 de los gradientes de la red
        output = red.forward(example.to(dispositivo))  # Propagacion hacia adelante
        loss = perdida_fn(output, label.to(dispositivo))  # Calculo de la perdida
        loss.backward()  # Propagacion hacia atras
        optimizador.step()  # Optimizacion de la pasada hacia atras


def valida_o_evalua(dispositivo, dataset, red, optimizador,
                     perdida_fn=nn.MSELoss(reduction='mean')):
    """
    Funcion que valida o testea la red neuronal pasada como parametro (tambien
    se deben pasar la funcion de perdida y el optimizador como parametros)
    sobre un dataset pasado com parametro. Esta funcion calcula además la
    precision y la perdida memdia de la red.

     @type dispositivo: str
    @param dispositivo: el dispositivo al que mandar los datos, en el que se
           encuentra el modelo

    @type dataset: DataLoader
    @param dataset: lista con las muestras y las etiquetas a enseñar a la red,
           distribuidas en batches

    @type red: ResNet
    @param red: modelo de la red neuronal ResNet

    @type optimizador: Optimizer
    @param optimizador: el tipo de algoritmo de descenso gradiente a utilizar

    @type perdida_fn: Loss
    @param perdida_fn: funcion de perdida a utilizar para ensenhar a la red
    """

    # Ponemos la red en modo evaluacion
    red.eval()

    prec = 0.0  # Precision de la red
    perd_med = 0.0  # Perdida media de la red

    # Recorremos todos los ejemplos con sus etiquetas
    for example, label in dataset:
        optimizador.zero_grad()  # Puesta a 0 de los gradientes de la red
        output = red.forward(example.to(dispositivo))  # Propagacion hacia adelante
        loss = perdida_fn(output, label.to(dispositivo))  # Calculo de la perdida

        guess = torch.max(output, -1)[1]  # Solucion que elige la red
        true_val = torch.max(label, -1)[1]  # Solucion correcta al ejemplo

        # Si coinciden, anhadir uno a la precision
        if guess.item() == true_val.item():
            prec += 1

        perd_med += loss.item()  # Acumulacion de la perdida

    # Calculamos la precision media y la perdida media y las devolvemos
    prec = prec / len(dataset)
    perd_med = perd_med / len(dataset)

    return [prec, perd_med]
