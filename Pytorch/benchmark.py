import os
import sys
from datetime import datetime

import numpy

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

# Lista de los dispositivos disponibles a utilizar por el programa
dispositivos = []
for i in range(0, torch.cuda.device_count()):
    dispositivos.append('cuda:' + str(i))


def inicializa_mundo(rank, world_size):
    """
    Funcion que inicializa el comunicador del grupo de procesos

    @type rank: int
    @param rank: rango del proceso actual

    @type world_size: int
    @param world_size: numero de procesos en el grupo de procesos
    """

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def limpieza_mundo():
    """
    Funcion que destruye el comunicador de procesos tras acabar la tarea
    en paralelo
    """
    dist.destroy_process_group()


def carga_datos():
    """
    Funcion que carga en listas los datos de los archivos binarios
    """

    with open('Entrenamiento/data.pkl', 'rb') as archivo_datos:
        datos = pk.load(archivo_datos)
    with open('Entrenamiento/labels.pkl', 'rb') as archivo_etiquetas:
        etiquetas = pk.load(archivo_etiquetas)

    return [datos, etiquetas]


def crea_tensores(parametros, datos, etiquetas, rango, tam_mundo):
    """
    Funcion que convierte los datos de las listas de muestras y etiquetas
    en datasets y dataloaders de Pytorch, el tamaño de estos depende
    del modo que se este usando, y la cantidad de datos que se quieran
    mandar a la red.

    @type parametros: dict
    @param parametros: Diccionario que contiene los tamanhos de los diferentes conjuntos

    @type datos: list
    @param datos: Lista con todas las muestras a ensenhar a la RN

    @type etiquetas: list
    @param etiquetas: Lista con las etiquetas de las muestras (a que clase pertenecen)

    @type rango: int
    @param rango: Rango del proceso actual

    @type tam_mundo: int
    @param tam_mundo: Tamanho del grupo de procesos
    """

    tam_entren = int(parametros.get('Tam. entrenamiento') * len(datos))
    tam_val = int(parametros.get('Tam. validacion') * len(datos)) + tam_entren

    # Ajuste del tamanho del conjunto de entrenamiento en caso de que se
    # seleccione el modo de operacion que dsitribuye las muestras
    # entre los dispositivos disponibles
    if parametros.get('Modo de operacion'):
        tam_entren = tam_entren / tam_mundo

        # Definicion de los tensores y dataloaders de la fase de entrenamiento
        datos_entren = torch.Tensor(datos[tam_entren * rango:tam_entren * (rango + 1)])
        etiquetas_entren = torch.Tensor(etiquetas[tam_entren * rango:tam_entren * (rango + 1)])

    else:
        # Definicion de los tensores y dataloaders de la fase de entrenamiento
        datos_entren = torch.Tensor(datos[:tam_entren])
        etiquetas_entren = torch.Tensor(etiquetas[:tam_entren])

    dataset_entren = TensorDataset(datos_entren, etiquetas_entren)
    loader_entren = DataLoader(dataset_entren, batch_size=parametros.get('Tam. batch'),
                               shuffle=True)

    # Definicion de los tensores y dataloaders de la fase de validacion
    datos_val = torch.Tensor(datos[tam_entren:tam_val])
    etiquetas_val = torch.Tensor(etiquetas[tam_entren:tam_val])

    dataset_val = TensorDataset(datos_val, etiquetas_val)
    loader_val = DataLoader(dataset_val, shuffle=True)

    # Definicion de los tensores y dataloaders de la fase de evaluacion
    datos_test = torch.Tensor(datos[tam_val:])
    etiquetas_test = torch.Tensor(etiquetas[tam_val:])

    dataset_test = TensorDataset(datos_test, etiquetas_test)
    loader_test = DataLoader(dataset_test, shuffle=True)

    return [loader_entren, loader_val, loader_test]


def entrenamiento_resnet(rank, world_size, parametros, datos):
    """
    Funcion que ejecuta en paralelo los bucles de entrenamiento, validacion y evaluacion
    de la red neuronal, para ello recibe como parametros el rango del proceso
    actual y el numero de procesos totales realizando la tarea.
    Los otros parametros recibidos se propagan a la funcion de creacion de los
    dataloaders con los que alimentar a la red neuronal.

    @type rank: int
    @param rank: rango del proceso actual

    @type world_size: int
    @param world_size: numero de procesos en el grupo de procesos

    @type parametros: dict
    @param parametros: Diccionario que contiene los tamanhos de los diferentes conjuntos

    @type datos: list
    @param datos: Lista con todas las muestras a ensenhar a la RN
    """

    # CONVERSION DE LOS DATOS AL FORMATO DE PYTORCH (TENSORES/DATALOADERS)
    loader_entren, loader_val, loader_test = crea_tensores(parametros, datos[0], datos[1], rank, world_size)

    # DEFINICION DE LA RED NEURONAL, OPTIMIZADOR Y PERDIDA

    # Inicializacion del comunicador de procesos, previo al inicio del
    # entrenamiento del modelo
    inicializa_mundo(rank, world_size)

    # Instancia de ResNet34 enviada al dispositivo designado al proceso
    # que ejecuta la funcion
    modelo = md.resNet34(in_channels=1, n_classes=9).to(dispositivos[rank])

    # Definicion del modelo distribuido (esto indica a Pytorch que hay una
    # copia del modelo por cada subproceso del grupo de procesos)
    modelo_paralelo = DDP(modelo)

    # No mandamos la funcion de perdida a ningun dispositivo,
    # ya que los resultados de la ejecucion en paralelo se unen en todos los
    # dispositivos mediante all reduce
    perdida_fn = torch.nn.MSELoss(reduction='mean')

    # Optimizador de la red (Stochastic Gradient Deescent)
    optimizador = torch.optim.SGD(modelo_paralelo.parameters(), lr=0.128)

    # ENTRENAMIENTO Y VALIDACION DE LA RED
    precisiones_epochs = []
    st = tm.time()
    for a in range(0, 24):
        tn.train(dispositivos[rank], loader_entren, modelo_paralelo, optimizador, perdida_fn)
        precision, = tn.validate_or_test(dispositivos[rank], loader_val, modelo_paralelo, optimizador, perdida_fn)
        precisiones_epochs.append(precision)

    t_entren = tm.time() - st

    # EVALUACION DE LA RED
    st = tm.time()
    precision, perdida = tn.validate_or_test(dispositivos[rank], loader_test, modelo_paralelo, optimizador)
    t_test = tm.time() - st

    # Eliminacion del grupo de procesos, operacion paralela terminada
    limpieza_mundo()

    # Manera de retornar al proceso principal los resultados de la ejecucion, mediante
    # la escritura de un archivo temporal, que el proceso principal lee al
    # terminar la ejecucion de esta funcion
    if dispositivos[rank] == 'cuda:0':
        with open('.cuda:0', 'wb') as archivo_temp:
            pk.dump([t_entren, t_test, precision, perdida, precisiones], archivo_temp)


def paraleliza_funcion(funcion, num_dispositivos, parametros, datos):
    """
    Funcion que ejecuta una funcion, pasada como parametro, en paralelo. Para
    ello crea tantos procesos como los especificados en el parametro
    'world_size'.
    El resto de parametros se le pasan a la funcion paralelizada.

    @type funcion: function
    @param funcion: Funcion a ejecutar en paralelo

    @type num_dispositivos: int
    @param num_dispositivos: Numero de procesos a crear para ejecutar la funcion en paralelo

    @type parametros: dict
    @param parametros: Diccionario que contiene los tamanhos de los diferentes conjuntos

    @type datos: list
    @param datos: Lista con todas las muestras a ensenhar a la RN
    """
    mp.spawn(funcion,
             args=(num_dispositivos, parametros, datos),
             nprocs=num_dispositivos,
             join=True)


if __name__ == '__main__':
    """
    Funcion principal de la script. Recibe como parametros (de la linea de
    comandos) el tamanho  de los conjuntos de de entrenamiento, validacion,
    tamanho de los batches del conjunto de entrenamiento y como distribuir
    los datos entre los dispositivos (0 = todos los dispositivos ven los mismos datos,
    1 = cada dispositivo ve una fraccion de los datos totales).
    
    Estos parametros deben introducirse en el orden especificado anteriormente,
    en caso de faltar alguno, se utilizaran valores por defecto para los
    parametros no intorducidos.
    """

    # Parametros por defecto
    tamanho_entren = 0.7
    tamanho_val = 0.2
    tamanho_batch = 16
    dividir_conjuntos = 0

    # Lectura de los parametros desde la linea de comandos
    try:
        tamanho_entren = sys.argv[1]
        tamanho_val = sys.argv[2]
        tamanho_test = sys.argv[3]
        tamanho_batch = sys.argv[4]
        dividir_conjuntos = sys.argv[5]

    except IndexError:
        print("No se han introducido todos los parametros esperados," +
              " usando valores por defecto")

    # Diccionario con los parametros con los que se ha ejecutado el benchamrk
    params = {'Tam. entrenamiento': tamanho_entren, 'Tam. validacion': tamanho_val,
              'Tam. batch': tamanho_batch, 'Modo de operacion': dividir_conjuntos,
              'Dispositivos': dispositivos}

    # Lectura de las muestras y etiquetas desde sus respectivos archivos
    lista_datos = carga_datos()

    # Lista con los resultados obtenidos
    resultados = []

    # Bucle del benchmark, que ejecuta el entrenamiento y evaluacion de la red
    # neuronal, distribuyendolo entre los dispositivos disponibles (GPUs) y
    # cogiendo los resultados que devuelve el dispositivo 'cuda:0' (los resul-
    # tados son iguales para todos los dispositivos)
    for i in range(0, 10):
        print('====RUN ' + str(i) + '====')
        paraleliza_funcion(entrenamiento_resnet, len(dispositivos), params, lista_datos)

        with open('.cuda:0', 'rb') as f:
            resultados.append(pk.load(f))

    # Agrupacion los resultados obtenidos de las x runs del benchmark en una
    # matriz, en la que cada fila es un tipo de resultado (t. entren,
    # t. val...) y cada columna es una run
    os.remove('.cuda:0')

    # Guardado de la evolucion de las precisiones a lo largo de los epochs
    precisiones = []
    for run in resultados:
        precisiones.append(run.pop())

    resultados = np.transpose(np.array(resultados))

    # Guardado de los resultados en un archivo, con la fecha y hora en la que
    # se termino el benchamrk
    with open(datetime.now().strftime('%Y-%m-%d,%H:%M:%S') + '.bbr', 'wb') as f:
        pk.dump([params, resultados], f)

    # Guardado de las precisiones de cada run
    numpy.savetxt('Precisiones ' + datetime.now().strftime('%Y-%m-%d,%H:%M:%S') + '.csv',
                  np.array(precisiones), delimiter=',')
