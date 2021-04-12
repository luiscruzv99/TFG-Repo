import os
import sys
from datetime import datetime

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

# Lista de los dispositivos a utilizar por el programa
# (anhadimos el cpu manualmente)
devices = ['cpu']
for i in range(0, torch.cuda.device_count()):
   devices.append('cuda:'+str(i))


def setup(rank, world_size):
    '''
    Funcion que inicializa el grupo de procesos,
    '''
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    '''
    Funcion que destruye el grupo de procesos tras acabar la tarea en paralelo
    '''
    dist.destroy_process_group()


def load_data():
    '''
    Funcion que carga en arrays los datos de los archivos pickle
    '''
    with open('Entrenamiento/data.pkl', 'rb') as f:
        data = pk.load(f)
    with open('Entrenamiento/labels.pkl', 'rb') as f:
        labels = pk.load(f)

    return [data, labels]


def convert_data(data, labels, rank, train_size, val_size ,batch_size, cpu_factor):
    '''
    Funcion que convierte de los arrays en datasets y dataloaders
    de Pytorch, el tamaño de estos depende del dispositivo que se
    este usando, y la cantidad de datos que se quieran mandar a 
    la red
    '''

       
    # Tamanho del dataset de entrenamiento del dispositivo, la cantidad de
    # datos que ve el modelo es train_size*world_size (#. de dispositivos).
    
    if(devices[rank] == 'cpu'):
        train_size = int(train_size / cpu_factor)

       
    # Definicion de los tensores y dataloaders de la fase de entrenamiento
    train_data = torch.Tensor(data[train_size * rank:train_size *
                                   (rank+1)])
    train_labels = torch.Tensor(labels[train_size * rank:train_size *
                                       (rank+1)])

    train_set = TensorDataset(train_data, train_labels)

    if(devices[rank] == 'cpu'):
        train_loader = DataLoader(train_set, batch_size=int(batch_size/cpu_factor), shuffle=True)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # Definicion de los tensores y dataloaders de la fase de validacion
    val_data = torch.Tensor(data[train_size: train_size+val_size])
    val_labels = torch.Tensor(labels[train_size:train_size+val_size])

    val_set = TensorDataset(val_data, val_labels)
    val_loader = DataLoader(val_set, shuffle=True)

    return [train_loader, val_loader]


def loop(rank, world_size, train_size, val_size, batch_size, cpu_factor):
    '''
    Funcion que ejecuta en paralelo los bucles de entrenamiento y validacion de
    la red neuronal, para ello recibe como parametros el rango del proceso
    actual y el numero de procesos totales realizando la tarea
    '''


    # CARGADO DE LOS DATOS DESDE LOS ARCHIVOS
    data, labels = load_data()
    
    # CONVERSION DE LOS DATOS AL FORMATO DE PYTORCH (TENSORES/DATALOADERS)
    train_loader, val_loader = convert_data(data=data, labels=labels,
                                            rank=rank, train_size=train_size,
                                            val_size=val_size, batch_size=batch_size,
                                            cpu_factor=cpu_factor)

    # Definicion de la red neuronal, optimizador y perdida

    # Inicializacion del grupo de procesos, previo al inicio del entrenamiento
    # y validacion del modelo
    setup(rank, world_size)

    # Instancia de ResNet34 enviada al dispositivo designado al proceso
    # que ejecuta la funcion
    model = md.resNet34(in_channels=1, n_classes=2).to(devices[rank])

    # Definicion del modelo distribuido (esto indica a Pytorch que hay una
    # copia del modelo por cada subproceso del grupo de procesos)
    net = DDP(model)

    # No mandamos la funcion de perdida a ningun dispositivo,
    # ya que los resultados de la ejecucion en paralelo se unen secuencialmente
    loss = torch.nn.MSELoss(reduction='mean')

    # Optimizador de la red (Stochastic Gradient Deescent)
    optimizador = torch.optim.SGD(net.parameters(), lr=0.128)

    # ENTRENAMIENTO DE LA RED
    t_entren = 0
    with net.join():
        st = tm.time()
        tn.train(devices[rank], train_loader, net, optimizador)
        t_entren = tm.time()-st

    # VALIDACION DE LA RED
    t_val = 0
    st = tm.time()
    acc, loss = tn.validate_or_test(devices[rank], val_loader,
                                    net, optimizador)

    t_val = tm.time()-st

    # Eliminacion del grupo de procesos, operacion paralela terminada
    cleanup()

    if(devices[rank] == 'cuda:0'):
        with open('.'+devices[rank], 'wb') as f:
            pk.dump([t_entren, t_val, acc, loss], f)


def run_loop(loop_fn, world_size,train_size, val_size, batch_size, cpu_factor):
    '''
    Funcion que ejecuta una funcion, pasada como parametro, en paralelo. Para
    ello spawnea tantos procesos como los especificados en el parametro
    'world_size'.
    '''
    mp.spawn(loop_fn,
            args=(world_size, train_size, val_size, batch_size, cpu_factor, ),
             nprocs=world_size,
             join=True)


'''
Funcion principal del script, punto de entrada para el entrenamiento de la red
neuronal. Para ello, se le especifica a run_loop la funcion que entrena y
valida la red (loop), con un tamanho de mundo (numero de procesos) igual al
numero de dispositivos disponibles en la maquina (incluyendo el procesador).
'''
if __name__ == '__main__':

    train_size = 50
    val_size = 10
    batch_size = 25
    cpu_factor = 5
    params = {'Tam. entrenamiento': train_size, 'Tam. validacion': val_size,
            'Tam. batch': batch_size, 'Factor reduccion CPU': cpu_factor}
    results = []
    try:
      train_size = sys.argv[1]
      val_size = sys.argv[2]
      batch_size = sys.argv[3]
      cpu_factor = sys.argv[4]
    except:
        print("No se han introducido todos los parametros esperados," +
            " usando valores por defecto")
    for i in range(0,3):
        print('====RUN '+str(i)+'====')
        run_loop(loop, len(devices), int(train_size), int(val_size), int(batch_size), int(cpu_factor))
        with open('.cuda:0', 'rb') as f:
            results.append(pk.load(f))
    
    os.remove('.cuda:0')
    ordered_results = np.transpose(np.array(results))
    with open(datetime.now().strftime('%Y %m %d, %H:%M:%S'), 'wb') as f:
        pk.dump([params, ordered_results], f)
