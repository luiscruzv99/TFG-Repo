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

# Lista de los dispositivos a utilizar por el programa (anhadimos el cpu manualmente)
devices = ['cpu']
for i in range(0, torch.cuda.device_count()):
    devices.append('cuda:'+str(i))


'''
Funcion que inicializa el grupo de procesos,
'''
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("gloo", rank=rank, world_size=world_size)


'''
Funcion que destruye el grupo de procesos tras acabar la tarea en paralelo
'''
def cleanup():
    dist.destroy_process_group()


'''
Funcion que detecta que dispositivo puede usar pytorch para entrenar la red
(No usado ya que utilizamos todos los dispositivos)
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
    end1 = 1000  # Tamanho del dataset de entrenamiento
    end2 = 25  # Tamanho del dataset de validacion
    
    '''
    Intento de equilibrado de carga, modificando el tamanho de los datasets para el CPU
    (No funciona de momento)
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


'''
Funcion que ejecuta en paralelo los bucles de entrenamiento y validacion de 
la red neuronal, para ello recibe como parametros el rango del proceso actual
y el numero de procesos totales realizando la tarea
'''
def loop(rank, world_size):
    print(f"Usando dispositivo {rank}: " + str(devices[rank]))  # Indicacion de que dispositivo esta realizando el trabajo 
    
    st = tm.time()
    data, labels = load_data()
    print("Cargado de los datos: "+str(tm.time()-st))

    st = tm.time()
    train_loader, val_loader = convert_data(data=data, labels=labels, device=devices[rank])
    print(f"{rank}: Creacion de datasets: "+str(tm.time()-st))

    # Definicion de la red neuronal, optimizador y perdida
    setup(rank, world_size)  #Inicializacion del grupo de procesos, previo al inicio del entrenamiento y validacion del modelo
    model = md.resNet34(in_channels=1,n_classes=2).to(devices[rank])  # Instancia de ResNet34 enviada al dispositivo designado al proceso que ejecuta la funcion
    net = DDP(model)  # Definicion del modelo distribuido (esto indica a Pytorch que hay una cpoia del modelo por cada subproceso del grupo de procesos)
    loss = torch.nn.MSELoss(reduction='mean')  # No mandamos la funcion de perdida a ningun dispositivo, ya que los resultados de la ejecucion en paralelo se unen secuencialmente
    optimizador = torch.optim.SGD(net.parameters(), lr=0.128)  # Optimizador de la red (Stochastic Gradient Deescent)
    
    # Barrera para sincronizar los procesos antes del inicio del entrenamiento
    dist.barrier()
    
    # Fase de entrenamiento 
    st = tm.time()
    tn.train(train_loader, net, optimizador)
    print(f"{rank}: Tiempo de entrenamiento: "+str(tm.time()-st))
    
    # Barrera para sincronizar los procesos antes del inicio de la validacion
    dist.barrier()
    
    # Fase de validacion
    st = tm.time()
    acc, loss = tn.validate_or_test(val_loader, net, optimizador)
    print(f"{rank}: Tiempo de validacion: "+str(tm.time()-st))
    
    cleanup()  # Eliminacion del grupo de procesos, operacion paralela terminada
    
    print("Precision: "+str(acc))
    print("Perdida: "+str(loss))
    

'''
Funcion que ejecuta una funcion, pasada como parametro, en paralelo. Para ello spawnea tantos procesos
como los especificados en el parametro 'world_size'.
'''
def run_loop(loop_fn, world_size):
    mp.spawn(loop_fn,
            args=(world_size, ),
            nprocs=world_size,
            join=True)


'''
Funcion principal del script, punto de entrada para el entrenamiento de la red neuronal.
Para ello, se le especifica a run_loop la funcion que entrena y valida la red (loop), con
un tamanho de mundo (numero de procesos) igual al numero de dispositivos disponibles en la
maquina.
'''
if __name__ == '__main__':
    run_loop(loop, len(devices))
