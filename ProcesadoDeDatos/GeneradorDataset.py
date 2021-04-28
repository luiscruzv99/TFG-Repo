import sys
import os
import time as t
import random

from PIL import Image as im
import numpy as np
import pickle as pk



def lee_datos(directorio, etiqueta):
    '''
     Esta funcion lee las imagenes de un directorio pasado como parametro y las convierte
     a arrays de numpy con valores normalizados (entre 0 y 1) que va almacenando en un array
     que se retorna al finalizar.
    '''

    imagenes = []  # Array que contiene todas las imagenes procesadas como arrays de numpy

    #st = t.time()

    # Recorrido de todos los archivos del directorio
    for archivo in os.listdir(directorio):

        if archivo.endswith('.png'):  # Cogemos solo los archivos .jpg
            # Lectura de una imagen (en b/n) y transformacion en un array de numpy
            imagen = im.open(os.path.join(directorio, archivo)).convert('L')
            # imagen = imagen.resize((344, 256))
            temp = (np.array(imagen, dtype=np.float16)/255)
            imagenes.append(np.array([temp]))

            imagen.save(str(archivo))

    # print("Tiempo lectura: " + str(t.time()-st))
    # print([etiqueta, len(imagenes)])
    return [imagenes, etiqueta]


def guarda_datos(directorio, datos, etiquetas):
    '''
     Esta funcion recibe un directorio destino y dos arrays de numpy y exporta dichos arrays
     a un archivo .pkl cada uno.
    '''

    with open(os.path.join(directorio,'data.pkl'), 'wb') as f:
        pk.dump(datos, f)

    with open(os.path.join(directorio, 'labels.pkl'), 'wb') as f:
        pk.dump(etiquetas, f)


if (__name__ == '__main__'):
    '''
    ===== PROGRAMA PRINCIPAL=====
    '''

    datos = []  # Imagenes de los directorios procesadas
    datos_aleatorios = [] # Imagenes de los directorios, distribidas aleatoriamente
    etiquetas_aleatorias = []  # Etiquetas de las imagenes, distribidas aleatoriamente

    try:
        clases = len(sys.argv)  # Numero de argumentos pasados al script

        # Comprobamos que se han pasado parametros
        if (clases == 1):
            raise IndexError()

        # Cogemos todas las imagenes de cada uno de los directorios pasados como
        # parametro y las convertimos en matrices bidimensionales normalizadas
        # (Las transformamos a imagenes blanco y negro, leemos su valor de brillo,
        # como de blancas son, y horquillamos dicho valor entre 0 y 1)
        for i in range(1, clases):
            datos.append(lee_datos(r'' + sys.argv[i],i-1))

        # Para poder tener sets de entrenamiento, validacion y test justos, hay que
        # distribuir de manera aleatoria las muestras
        # Mientras haya cosas en datos
        # TODO: ARREGLA ESTO GILIPOLLAS
        while(datos):
            # Genera un num aleatorio entre 0 y len(datos-1)
            entrada = random.randint(0, len(datos)-1)
            # Coge la lista de imagenes de la entrada de la lista datos
            # Quita el primer elemento de dicha entrada
            # Anhade ese elemento a la lista de datos_aleatorios
            datos_aleatorios.append(datos[entrada][0].pop())
            # Anhade el numero aleatorio generado antes a la lista de etiquetas_aleatorias
            etiqueta = []
            num_etiqueta = datos[entrada][1]
            for i in range(0, clases-1):
                if (i == num_etiqueta): etiqueta.append(1)
                else: etiqueta.append(0)
            etiquetas_aleatorias.append(etiqueta)
            if(len(datos[entrada][0]) == 0): datos.remove(datos[entrada])


        # Una vez reorganizadas las muestras, las guardamos como un par de archivos
        # binarios que cargaremos mas tarde para alimentar al modelo de ResNet34
        guarda_datos('', datos_aleatorios, etiquetas_aleatorias)
        # print(datos_aleatorios[0])
        # print(datos_aleatorios[0].shape)

    except IndexError:
        print("Se esperaban argumentos indicando los directorios con las imagenes")
        sys.exit(2)
