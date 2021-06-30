import os
import sys
from datetime import datetime

import numpy
import pandas as pd
from tqdm import tqdm

import Modelo.ResNet as md
import Modelo.entrenamiento as tn

import pickle as pk
import numpy as np
import time as tm
import json

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from torch.utils.data import TensorDataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
def carga_datos():
    """
    Funcion que carga en listas los datos de los archivos binarios
    """

    with open('Datos/data.pkl', 'rb') as archivo_datos:
        datos = pk.load(archivo_datos)
    with open('Datos/labels.pkl', 'rb') as archivo_etiquetas:
        etiquetas = pk.load(archivo_etiquetas)

    return [datos, etiquetas]


def crea_tensores(parametros, datos, etiquetas):
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
    """

    tam_entren = int(parametros.get('Tam. entrenamiento') * len(datos))
    tam_val = int(parametros.get('Tam. validacion') * len(datos)) + tam_entren
    tam_test = int(parametros.get('Tam. test') * len(datos)) + tam_val

    # Definicion de los tensores y dataloaders de la fase de entrenamiento
    datos_entren = torch.Tensor(datos[0:tam_entren])
    etiquetas_entren = torch.Tensor(etiquetas[0:tam_entren])


    dataset_entren = TensorDataset(datos_entren, etiquetas_entren)
    loader_entren = DataLoader(dataset_entren, batch_size=parametros.get('Tam. batch'),
                               shuffle=True)
    # print(len(dataset_entren))
    # Definicion de los tensores y dataloaders de la fase de validacion
    datos_val = torch.Tensor(datos[tam_entren:tam_val])
    etiquetas_val = torch.Tensor(etiquetas[tam_entren:tam_val])

    dataset_val = TensorDataset(datos_val, etiquetas_val)
    loader_val = DataLoader(dataset_val, shuffle=True)

    # Definicion de los tensores y dataloaders de la fase de evaluacion
    datos_test = torch.Tensor(datos[tam_val:tam_test])
    etiquetas_test = torch.Tensor(etiquetas[tam_val:tam_test])

    dataset_test = TensorDataset(datos_test, etiquetas_test)
    loader_test = DataLoader(dataset_test, shuffle=True)

    return [loader_entren, loader_val, loader_test]


def entrenamiento_resnet(parametros, datos, dispositivo):
    """
    Funcion que ejecuta en paralelo los bucles de entrenamiento, validacion y evaluacion
    de la red neuronal, para ello recibe como parametros el rango del proceso
    actual y el numero de procesos totales realizando la tarea.
    Los otros parametros recibidos se propagan a la funcion de creacion de los
    dataloaders con los que alimentar a la red neuronal.

    @type parametros: dict
    @param parametros: Diccionario que contiene los tamanhos de los diferentes conjuntos

    @type datos: list
    @param datos: Lista con todas las muestras a ensenhar a la RN

    @type dispositivo: str
    @param dispositivo: dispositivo en el que ejecutar el entrenamiento
    """

    # CONVERSION DE LOS DATOS AL FORMATO DE PYTORCH (TENSORES/DATALOADERS)
    loader_entren, loader_val, loader_test = crea_tensores(parametros, datos[0], datos[1])

    # DEFINICION DE LA RED NEURONAL, OPTIMIZADOR Y PERDIDA

    # Instancia de ResNet34 enviada al dispositivo designado al proceso
    # que ejecuta la funcion
    modelo = md.resNet34(in_channels=1, n_classes=9).to(dispositivo)

    # No mandamos la funcion de perdida a ningun dispositivo,
    # ya que los resultados de la ejecucion en paralelo se unen en todos los
    # dispositivos mediante all reduce
    perdida_fn = torch.nn.MSELoss(reduction='mean')

    # Optimizador de la red (Stochastic Gradient Deescent)
    optimizador = torch.optim.SGD(modelo.parameters(), lr=0.05)

    # ENTRENAMIENTO Y VALIDACION DE LA RED
    precisiones_epochs = []
    t_entrenamiento_epochs = []
    t_validacion_epochs = []
    st = tm.time()

    for _ in range(100):
        ste = tm.time()
        tn.entrena(dispositivo, loader_entren, modelo, optimizador, perdida_fn)
        t_entrenamiento_epochs.append(tm.time()-ste)
        ste = tm.time()
        precision, perdida = tn.valida_o_evalua(dispositivo, loader_val, modelo, optimizador, perdida_fn)
        t_validacion_epochs.append(tm.time()-ste)

        precisiones_epochs.append(precision)

    t_entren = tm.time() - st

    # EVALUACION DE LA RED
    st = tm.time()
    precision, perdida = tn.valida_o_evalua(dispositivo, loader_test,
                                             modelo, optimizador)
    t_test = tm.time() - st

    return [t_entren, t_test, precision, perdida, precisiones_epochs, t_entrenamiento_epochs, t_validacion_epochs]


def main():
    """
       Funcion principal de la script. Recibe como parametro (de la linea de
       comandos) el nombre del fichero.

       Ejecuta una run del benchmark, y guarda los resultados obtenidos en un fichero
       binario.
       """

    # Parametros por defecto
    tamanho_entren = 0.85
    tamanho_val = 0.02
    tamanho_test = 1 - (tamanho_entren + tamanho_val)
    tamanho_batch = 128

    # Lectura del parametro desde la linea de comandos
    try:
        nombre_fich = sys.argv[1]
        dispositivo = sys.argv[2]

    except IndexError:
        print("No se han introducido todos los parametros, saliendo")
        sys.exit(2)

    # Diccionario con los parametros con los que se ha ejecutado el benchamrk
    params = {'Tam. entrenamiento': tamanho_entren, 'Tam. validacion': tamanho_val,
              'Tam. test': tamanho_test, 'Tam. batch': tamanho_batch, 'Dispositivo': dispositivo}

    # Lectura de las muestras y etiquetas desde sus respectivos archivos
    lista_datos = carga_datos()

    # Bucle del benchmark, que ejecuta el entrenamiento y evaluacion de la red
    # neuronal en el dispositivo especificado
    resultados = entrenamiento_resnet(params, lista_datos, dispositivo)

    # Guardado de la evolucion de la red a lo largo de los epochs
    validaciones = resultados.pop()
    entrenamientos = resultados.pop()
    precisiones = resultados.pop()

    # Formateo de la lista de parametros, para facilitar su legibilidad
    params['Tam. muestras'] = lista_datos[0][0][0].shape
    params['Tam. entrenamiento'] = int(params['Tam. entrenamiento'] * len(lista_datos[0]))
    params['Tam. validacion'] = int(params['Tam. validacion'] * len(lista_datos[0]))
    params['Tam. test'] = int(params['Tam. test'] * len(lista_datos[0]))

    # Guardado de los resultados en un archivo temporal, con el nombre
    # especificao en la linea de comandos
    with open(nombre_fich, 'wb') as f:
        pk.dump([resultados, params, validaciones, entrenamientos, precisiones], f)

if __name__ == '__main__':
   main()
