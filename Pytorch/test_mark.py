import os

import Modelo.model as md
import Modelo.training as tn

import pickle as pk
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


def convert_data(data, labels, rank):
    '''
    Funcion que convierte de los arrays en tensores de pytorch, mandandolos a
    la GPU, definicion de datasets y dataloaders
    '''

    # Tamanho del dataset de entrenamiento del dispositivo, la cantidad de
    # datos que ve el modelo es train_size*world_size (#. de dispositivos).
    train_size = 3500

    # Tamanho del dataset de validacion, este es independiente del dispositivo.
    # val_size = 25

    # Definicion de los tensores y dataloaders de la fase de entrenamiento
    train_data = torch.Tensor(data[train_size * rank:train_size *
                                   (rank+1)])
    train_labels = torch.Tensor(labels[train_size * rank:train_size *
                                       (rank+1)])

    train_set = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_set, batch_size=24, shuffle=True)

    # Definicion de los tensores y dataloaders de la fase de validacion
    val_data = torch.Tensor(data[30000:30500])
    val_labels = torch.Tensor(labels[30000:30500])

    val_set = TensorDataset(val_data, val_labels)
    val_loader = DataLoader(val_set, shuffle=True)

    return [train_loader, val_loader]


def loop(rank, world_size):
    '''
    Funcion que ejecuta en paralelo los bucles de entrenamiento y validacion de
    la red neuronal, para ello recibe como parametros el rango del proceso
    actual y el numero de procesos totales realizando la tarea
    '''

    # Indicacion de que dispositivo esta realizando el trabajo
    print(f"Usando dispositivo {rank}: " + str(devices[rank]))

    # CARGADO DE LOS DATOS DESDE LOS ARCHIVOS
    st = tm.time()
    data, labels = load_data()
    print("Cargado de los datos: "+str(tm.time()-st))

    # CONVERSION DE LOS DATOS AL FORMATO DE PYTORCH (TENSORES/DATALOADERS)
    st = tm.time()
    train_loader, val_loader = convert_data(data=data, labels=labels,
                                            rank=rank)

    print(f"{rank}: Creacion de datasets: "+str(tm.time()-st))

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

    # Barrera para sincronizar los procesos antes del inicio del entrenamiento
    # dist.barrier()

    # ENTRENAMIENTO DE LA RED
    st = tm.time()
    tn.train(devices[rank], train_loader, net, optimizador)
    print(f"{rank}: Tiempo de entrenamiento: "+str(tm.time()-st))

    # Barrera para sincronizar los procesos antes del inicio de la validacion
    # dist.barrier()

    # VALIDACION DE LA RED
    st = tm.time()
    acc, loss = tn.validate_or_test(devices[rank], val_loader,
                                    net, optimizador)

    print(f"{rank}: Tiempo de validacion: "+str(tm.time()-st))

    # Eliminacion del grupo de procesos, operacion paralela terminada
    cleanup()

    print("Precision: "+str(acc))
    print("Perdida: "+str(loss))


def run_loop(loop_fn, world_size):
    '''
    Funcion que ejecuta una funcion, pasada como parametro, en paralelo. Para
    ello spawnea tantos procesos como los especificados en el parametro
    'world_size'.
    '''
    mp.spawn(loop_fn,
             args=(world_size, ),
             nprocs=world_size,
             join=True)


'''
Funcion principal del script, punto de entrada para el entrenamiento de la red
neuronal. Para ello, se le especifica a run_loop la funcion que entrena y
valida la red (loop), con un tamanho de mundo (numero de procesos) igual al
numero de dispositivos disponibles en la maquina (incluyendo el procesador).
'''
if __name__ == '__main__':
    run_loop(loop, len(devices))
