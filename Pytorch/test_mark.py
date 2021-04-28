import os
import sys
from datetime import datetime

import Modelo.model as md
import Modelo.training as tn

import pickle as pk
import numpy as np
import time as tm
import json

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
    Funcion que destruye el comunicador de procesos tras acabar la tarea
    en paralelo
    '''
    dist.destroy_process_group()


def load_data():
    '''
    Funcion que carga en arrays los datos de los archivos binarios
    '''
    with open('Entrenamiento/data.pkl', 'rb') as f:
        data = pk.load(f)
    with open('Entrenamiento/labels.pkl', 'rb') as f:
        labels = pk.load(f)

    return [data, labels]


def convert_data(data, labels, rank, train_size, val_size, batch_size,
                 cpu_factor):
    '''
    Funcion que convierte de los arrays en datasets y dataloaders
    de Pytorch, el tamaño de estos depende del dispositivo que se
    este usando, y la cantidad de datos que se quieran mandar a
    la red
    '''

    # Tamanho del dataset de entrenamiento del dispositivo. El CPU puede
    # recibir menos datos, para compensar su inferior capacidad de computo,
    # mediante el parametro cpu_factor.

    if(devices[rank] == 'cpu'):
        train_size = int(train_size / cpu_factor)

    # Definicion de los tensores y dataloaders de la fase de entrenamiento
    train_data = torch.Tensor(data[train_size * rank:train_size *
                                   (rank+1)])
    train_labels = torch.Tensor(labels[train_size * rank:train_size *
                                       (rank+1)])

    train_set = TensorDataset(train_data, train_labels)

    if(devices[rank] == 'cpu'):
        train_loader = DataLoader(train_set,
                                  batch_size=int(batch_size/cpu_factor),
                                  shuffle=True)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  shuffle=True)

    # Definicion de los tensores y dataloaders de la fase de validacion
    val_data = torch.Tensor(data[30000: 30000+val_size])
    val_labels = torch.Tensor(labels[30000:30000+val_size])

    val_set = TensorDataset(val_data, val_labels)
    val_loader = DataLoader(val_set, shuffle=True)

    return [train_loader, val_loader]


def loop(rank, world_size, train_size, val_size, batch_size, cpu_factor):
    '''
    Funcion que ejecuta en paralelo los bucles de entrenamiento y validacion de
    la red neuronal, para ello recibe como parametros el rango del proceso
    actual y el numero de procesos totales realizando la tarea.
    Los otros parametros recibidos se propagan a la funcion de creacion de los
    dataloaders con los que alimentar a la red neuronal.
    '''

    # CARGADO DE LOS DATOS DESDE LOS ARCHIVOS
    data, labels = load_data()

    # CONVERSION DE LOS DATOS AL FORMATO DE PYTORCH (TENSORES/DATALOADERS)
    train_loader, val_loader = convert_data(data=data, labels=labels,
                                            rank=rank, train_size=train_size,
                                            val_size=val_size,
                                            batch_size=batch_size,
                                            cpu_factor=cpu_factor)

    # DEFINICION DE LA RED NEURONAL, OPTIMIZADOR Y PERDIDA

    # Inicializacion del comunicador de procesos, previo al inicio del
    # entrenamiento y validacion del modelo
    setup(rank, world_size)

    # Instancia de ResNet34 enviada al dispositivo designado al proceso
    # que ejecuta la funcion
    model = md.resNet34(in_channels=1, n_classes=2).to(devices[rank])

    # Definicion del modelo distribuido (esto indica a Pytorch que hay una
    # copia del modelo por cada subproceso del grupo de procesos)
    net = DDP(model)

    # No mandamos la funcion de perdida a ningun dispositivo,
    # ya que los resultados de la ejecucion en paralelo se unen en todos los
    # dispositivos mediante all reduce
    loss = torch.nn.MSELoss(reduction='mean')

    # Optimizador de la red (Stochastic Gradient Deescent)
    optimizador = torch.optim.SGD(net.parameters(), lr=0.128)

    # ENTRENAMIENTO DE LA RED
    t_entren = 0
    # Esto evita que el entrenamiento se cuelgue si varios dispositivos ven
    # diferentes cantidades de datos
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

    # Manera de retornar al hilo principal los resultados de la run, mediante
    # la escritura de un archivo temporal, que el hilo principal lee al
    # terminar la ejecucion de esta funcion
    if(devices[rank] == 'cuda:0'):
        with open('.'+devices[rank], 'wb') as f:
            pk.dump([t_entren, t_val, acc, loss], f)


def run_loop(loop_fn, world_size, train_size, val_size, batch_size,
             cpu_factor):
    '''
    Funcion que ejecuta una funcion, pasada como parametro, en paralelo. Para
    ello spawnea tantos procesos como los especificados en el parametro
    'world_size'.
    El resto de parametros se le pasan a la funcion paralelizada.
    '''
    mp.spawn(loop_fn,
             args=(world_size, train_size, val_size, batch_size, cpu_factor, ),
             nprocs=world_size,
             join=True)


if __name__ == '__main__':
    '''
    Funcion principal de la script. Recibe como parametros (de la linea de
    comandos) el tamanho de entrenamiento y validacion de los conjuntos de
    datos,tamanho de los batches del conjunto de entrenamiento y la proporcion
    en la que reducir el tamanho de los datos de entrenamiento para la CPU
    (equilibrado de carga).
    Estos parametros deben introducirse en el orden especificado anteriormente,
    en caso de faltar alguno, se utilizaran valores por defecto para los
    parametros no intorducidos.
    '''

    # Parametros por defecto
    train_size = 3500
    val_size = 500
    batch_size = 24
    cpu_factor = 8

    # Lectura de los parametros desde la linea de comandos
    try:
        train_size = sys.argv[1]
        val_size = sys.argv[2]
        batch_size = sys.argv[3]
        cpu_factor = sys.argv[4]
    except Exception:
        print("No se han introducido todos los parametros esperados," +
              " usando valores por defecto")

    # Diccionario con los parametros con los que se ha ejecutado el benchamrk
    params = {'Tam. entrenamiento': train_size, 'Tam. validacion': val_size,
              'Tam. batch': batch_size, 'Factor reduccion CPU': cpu_factor,
              'Dispositivos': devices}

    # Lista con los resultados obtenidos
    results = []

    # Bucle del benchmark, que ejecuta el entrenamiento y validacion de la red
    # neuronal, distribuyendolo entre los dispositivos disponibles (CPU y GPUs)
    # cogiendo los resultados que devuelve el dispositivo 'cuda:0' (los resul-
    # tados son iguales para todos los dispositivos)
    for i in range(0, 10):
        print('====RUN '+str(i)+'====')

        run_loop(loop, len(devices), int(train_size), int(
            val_size), int(batch_size), int(cpu_factor))

        with open('.cuda:0', 'rb') as f:
            results.append(pk.load(f))

    # Agrupacion los resultados obtenidos de las x runs del benchmark en una
    # matriz, en la que cada fila es un tipo de resultado (t. entren,
    # t. val...) y cada columna es una run
    os.remove('.cuda:0')

    ordered_results = np.transpose(np.array(results))

    # Guardado de los resultados en un archivo, con la fecha y hora en la que
    # se termino el benchamrk
    with open('Results ' + datetime.now().strftime('%Y-%m-%d,%H:%M:%S')+'.bbr',
              'wb') as f:
        pk.dump([params, ordered_results], f)

    # Guardadado de los parametros de la ejecucion en un archivo json, por
    # legibilidad
    with open('Results ' + datetime.now().strftime('%Y-%m-%d,%H:%M:%S') +
              '.json', 'w') as f:
        json.dump(params, f)

    # Guardado de los resultados de la ejecucion en un archivo .csv, por
    # legibilidad
    np.savetxt('Results ' + datetime.now().strftime('%Y-%m-%d,%H:%M:%S') +
               '.csv', results, delimiter=',')
