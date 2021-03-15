import os
import sys
import tempfile

import Modelo.model as md
import Modelo.training as tn

import pickle as pk
import numpy as np
import time as tm

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from torch.utils.data import TensorDataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

# Anhadimos el cpu a la lista de dispositivos para poder enviarle trabajos
devices = ['cpu']
for i in range(0, torch.cuda.device_count()):
    devices.append('cuda:'+str(i))


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


'''
Funcion que detecta que dispositivo puede usar pytorch para entrenar la red

def check_device():

    if torch.cuda.is_available():
        device = torch.device('cuda')  # Entrenar con la GPU primaria
    else:
        device = torch.device('cpu')  # Entrenar con el CPU

    return device
'''

'''
Funcion que carga en arrays los datos de los archivos pickle
'''
def load_data():
    
    with open('Entrenamiento/data.pkl','rb') as f:
        data = pk.load(f)
    with open('Entrenamiento/labels.pkl','rb') as f:
        labels = pk.load(f)

    return [data, labels]


'''
Funcion que convierte de los arrays en tensores de pytorch, mandandolos a la GPU, definicion de datasets y dataloaders
'''
def convert_data(data, labels, device):
   
    #Distribucion de cantidad de trabajos basado en dispositivo (no funciona)
    end1 = 1000
    end2 = 25
    '''
    if(device == 'cpu'):
        end1 = 50
    '''

    # Definicion de los tensores y dataloaders de la fase de entrenamiento
    train_data = torch.Tensor(data[:end1]).to(device) 
    train_labels = torch.Tensor(labels[:end1]).to(device) 

    train_set = TensorDataset(train_data,train_labels)
    train_loader = DataLoader(train_set, batch_size=5, shuffle=True)

    # Definicion de los tensores y dataloaders de la fase de validacion 
    val_data = torch.Tensor(data[end1:end1+end2]).to(device)
    val_labels = torch.Tensor(labels[end1:end1+end2]).to(device)

    val_set = TensorDataset(val_data, val_labels)
    val_loader = DataLoader(val_set, shuffle=True)

    return [train_loader, val_loader]



# device = check_device()

def loop(rank, world_size):
    st = tm.time()
    data, labels = load_data()
    print("Cargado de los datos: "+str(tm.time()-st))


    print("Usando dispositivo " + str(devices[rank]))
    print(f"Trabajando en rango {rank}")
    st = tm.time()
    train_loader, val_loader = convert_data(data=data, labels=labels, device=devices[rank])
    print("Creacion de datasets: "+str(tm.time()-st))

    # Definicion de la red neuronal, optimizador y perdida
    setup(rank, world_size)
    model = md.resNet34(in_channels=1,n_classes=2).to(devices[rank])  # Instancia de ResNet34
    net = DDP(model) 
    loss = torch.nn.MSELoss(reduction='mean') 
    optimizador = torch.optim.SGD(net.parameters(), lr=0.128)  # Optimizador de la red (Stochastic Gradient Deescent)
    
    dist.barrier()
    # Fase de entrenamiento 
    st = tm.time()
    tn.train(train_loader, net, optimizador)
    print("Tiempo de entrenamiento: "+str(tm.time()-st))
    
    dist.barrier()
    # Fase de validacion
    st = tm.time()
    acc, loss = tn.validate_or_test(val_loader, net, optimizador)
    print("Tiempo de validacion: "+str(tm.time()-st))
    print("Precision: "+str(acc))
    print("Perdida: "+str(loss))
    cleanup()

def run_loop(loop_fn, world_size):
    mp.spawn(loop_fn,
            args=(world_size, ),
            nprocs=world_size,
            join=True)

if __name__ == '__main__':
    run_loop(loop, len(devices))
