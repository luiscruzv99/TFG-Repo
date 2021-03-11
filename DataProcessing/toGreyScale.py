import sys
import os
import time as t
import random

from PIL import Image as im
import numpy as np
import pickle as pk


'''
 Esta funcion lee las imagenes de un directorio pasado como parametro y las convierte
 a arrays de numpy con valores normalizados (entre 0 y 1) que va almacenando en un array
 que se retorna al finalizar.
'''
def read_data(directory, labelValue=0):

    data = []  # Array que contiene todas las imagenes procesadas como arrays de numpy

    #st = t.time()

    # Recorrido de todos los archivos del directorio
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):  # Cogemos solo los archivos .jpg

            # Lectura de una imagen (en b/n) y transformacion en un array de numpy
            img = im.open(os.path.join(directory, filename)).convert('L')  
            tmp = (np.array(img, dtype=np.float16)/255)
            data.append(np.array([tmp]))

            # test.save(os.path.join(directory,'bw/'+str(filename)))

    #print("Tiempo lectura: " + str(t.time()-st))

    return data

'''
 Esta funcion recibe un directorio destino y dos arrays de numpy y exporta dichos arrays
 a un archivo .pkl cada uno.
'''
def save_data(directory, data, labels):

    with open(os.path.join(directory,'data.pkl'), 'wb') as f:
        pk.dump(data, f)

    with open(os.path.join(directory, 'labels.pkl'), 'wb') as f:
        pk.dump(labels, f)


'''
===== PROGRAMA PRINCIPAL=====
'''

directory0=r''
directory1=r''

try:
    directory0 += sys.argv[1] # Directorio de la clase positiva
    directory1 += sys.argv[2] # Directorio de la clase negativa
except:
    print("Se esperaban dos argumentos indicando los directorios con las imagenes")
    sys.exit(2)


data = read_data(directory0, 1)
data1 = read_data(directory1)

data_batch = [data, data1]

random_data = []
random_labels = []

while data_batch[0] or data_batch[1]:

    d=[0,0]
    if not data_batch[0]:
        t = 1
    elif not data_batch[1]:
        t = 0
    else:
        t = random.randint(0,1)

    d[t] = 1
    e= random.randint(0, len(data_batch[t])-1)

    random_data.append(data_batch[t].pop(e))
    random_labels.append(d)

save_data('', random_data, random_labels)

print(random_data[0].shape)
