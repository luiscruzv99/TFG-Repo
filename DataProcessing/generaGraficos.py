import os
import sys

import pickle as pk
import pandas as pd
import matplotlib.pyplot as plt


def abrir_resultados(archivo):
    '''
    Funcion que abre el archivo pickle especificado como parametro
    y carga los datos que contiene, retornandolos

    El formato esperado de este archivo es una lista de python que
    contiene un diccionario, y una matriz de numpy

    @param archivo, nombre o ruta del archivo a leer
    @return datos, los datos que contiene el archivo
    '''

    with open(archivo, 'rb') as f:
        datos = pk.load(f)

    return datos


def genera_grafico(datos, etiqueta_x, etiqueta_y, etiqueta_y2, titulo,
                   directorio):
    '''
    Funcion que genera y guarda a un archivo un grafico mixto
    de barras y lineas en funcion de unos datos pasados
    como parametros

    @param datos, los datos a representar, se espera una matriz
    de tres filas
    @param etiqueta_x, etiqueta del eje x
    @param etiqueta_y, etiqueta del eje y del grafico de barras
    @param etiqueta_y2, etiqueta del eje y del grafico de linea
    @param titulo, titulo del grafico y nombre del archivo a
    guardar
    @param directorio, ruta donde guardar el archivo
    '''

    grafico = pd.DataFrame({
        'T. Entrenamiento': datos[0],
        'T. Validacion': datos[1],
        'Precision': datos[2]})

    # Esto genera el grafico de barras
    grafico[['T. Entrenamiento', 'T. Validacion']].plot(kind='bar',
                                                        width=0.5,
                                                        xlabel=etiqueta_x,
                                                        ylabel=etiqueta_y,
                                                        color=['red', 'blue'],
                                                        edgecolor='black')

    # Esto genera el grafico de lineas
    grafico['Precision'].plot(secondary_y=True, color='green', linewidth=2)

    # Esto anhade las etiquetas y titulo al grafico y lo
    # guarda en el directorio especificado
    plt.ylabel(etiqueta_y2)
    plt.title(titulo)
    plt.savefig(directorio + '/' + titulo + '.png')


if __name__ == '__main__':

    '''
    Punto de entrada de la script, recoge de consola el archivo
    a leer y con los datos del archivo genera un grafico y lo guarda
    '''

    try:
        # Cogemos el parametro de la consola, y si no hay ninguno, abortamos
        archivo = sys.argv[1]
        params, resultados = abrir_resultados(archivo)
        # Creamos un directorio con nombre igual al del archivo
        # Ahi es donde guardaremos el grafico
        os.mkdir(archivo[:-4])
        # Generamos el grafico, en este caso utilizando el tiempo de
        # entrenamiento,tiempo de validacion y precision de la red. Como titulo
        # ponemos los parametros utilizados para correr el benchmark (tamanho
        # entrenamiento, tamanho validacion, tamanho de los batches y factor de
        # reduccion del cpu)
        genera_grafico(resultados[:3], "Nº de run", "Tiempo (seg)",
                       "Precision en validacion (%)",
                       str(list(params.values())), archivo[:-4])
    except OSError:
        print("Error crendo directorio")
    except IndexError:
        print("Por favor, indica el archivo que quieres leer")
