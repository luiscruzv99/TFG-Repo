import os
import sys

import json

import numpy as np
import pandas as pd
import pickle as pk

from datetime import datetime

def main():
    """
    Funcion que coge cada uno de los resultados de los benchmarks, y los
    aglomera en archivos csv
    """

    archivos = 2  # Numero de ejecuciones
    general = []  # Resultados generales de la ejecución de los benchmarks
    parametros = {}  # Parametros de los benchmarks (no varian entre
                     # ejecuciones de la misma tanda)
    entrenamientos = []  # Tiempos de entrenamiento
    validaciones = []  # Tiempos de validacion
    precisiones = []  # Precisiones de los entrenamientos de los benchmarks

    # Para cada archivo
    for i in range(archivos):
        # Lo abrimos y cargamos sus contenidos
        with open(str(i), 'rb') as archivo:
             datos = pk.load(archivo)

        precisiones.append(datos.pop())
        entrenamientos.append(datos.pop())
        validaciones.append(datos.pop())
        parametros = datos.pop()
        general.append(datos.pop())
        # Eliminamos el archivo (no se necesita mas)
        os.remove(str(i))

    # Formateo de los resultados obtenidos, anhadiendo etiquetas para su
    # posterior guardado en un archivo csv
    general = np.transpose(general)
    general_dict = {'Tiempo Entrenamiento (s)': general[0],
                    'Tiempo Evaluacion (s)': general[1],
                    'Precision (%)': general[2], 'Error (%)': general[3]}

    general = pd.DataFrame.from_dict(general_dict)

    directorio = "Resultados-" + datetime.now().strftime('%Y-%m-%d')
    os.mkdir(directorio)

    # Guardado de los parametros utilizados en un formato legible por humanos
    with open(directorio +'/Parametros.json', 'w') as f:
        json.dump(parametros, f)

    # Guardado de los resultados de tiempo obtenidos en un formato legible
    # por humanos
    general.to_csv(directorio+'/General.csv')

    # Guardado de las precisiones de cada epoch de cada run en un formato
    # legible por humanos
    np.savetxt(directorio + '/Precisiones.csv',
                  np.array(precisiones), delimiter=',', fmt='%f')

    # Guardado de los tiempos de entrenamiento de cada epoch de cada run en
    # un formato legible por humanos
    np.savetxt(directorio + '/Entrenamientos.csv',
                  np.array(entrenamientos), delimiter=',', fmt='%f')

    # Guardado de los tiempos de validacion de cada epoch de cada run en
    # un formato legible por humanos
    np.savetxt( directorio + '/Validaciones.csv',
                  np.array(validaciones), delimiter=',', fmt='%f')


if __name__ == '__main__':
    main()
