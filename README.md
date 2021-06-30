# Léeme - TFG



## Descripción

Este repositorio contiene el Trabajo Fin de Grado de Luis Cruz Varona, en el que se examina la diferencia de rendimiento de implementaciones de redes neuronales en Pytorch en ambientes Mono-GPU y Multi-GPU. Para ello se encuentran hechos 3 benchmarks y 4 experimentos, que se sirven de un modelo ResNet34 y de los estándares definidos por [MLCommons](https://mlcommons.org/en/) en sus benchmarks de [Training](https://mlcommons.org/en/training-normal-07/).

### · Estructura del repositorio

El repositorio contiene 3 grandes carpetas:

- __ProcesadoDeDatos__: Carpeta que contiene el script que preprocesa el dataset utilizado por los benchmarks, para ahorrar tiempo de ejecución.
- __Pytorch__: Carpeta que contiene las implementaciones del modelo, bucles de entrenamiento, validación y test, y benchmarks utilizando diferentes tipos de dispositivos



## Requisitos

Para poder ejecutar los benchmarks, es necesario tener instalado el siguiente software:

-  __Python3__ y __Pip__: para instalar ambos en Ubuntu, hacer:

``` bash
sudo apt update
sudo apt install python3 python3-pip
```

- Librerías de __CUDA__ necesarias para la ejecución de ambos frameworks en GPU: en este caso __CUDA-toolkit__ y __libcudnn7__ además de los drivers de nvidia.


- El framework de __Pythorch__: debe ser instalado a través de un gestor de librerías de Python, en este caso, a través de `pip`:

```bash
pip3 install torch torchvision torchaudio
```
- El programa de perfilado __sauna__ para poder realizar las medidas de energía.





 ## Uso

### · Preprocesado del dataset

Para poder usar los benchmarks, es necesario preprocesar el dataset a usar. Se recomienda el uso del dataset utilizado en el proyecto: [A Large Scale Fish Dataset](https://www.kaggle.com/crowww/a-large-scale-fish-dataset). Para prepocesar el dataset se puede usar el script de python `ProcesadoDeDatos/GeneradorDataset.py`. Este script recibe como parámetros el directorio de cada una de las clases del dataset (1 directorio = 1 clase) y crea dos archivos binarios `data.pkl` y `labels.pkl`.

Ejemplo de uso:

```bash
python3 GeneradorDataset.py dataset/clase1/ dataset/clase2/ dataset/clase3/
```



### · Lanzamiento de los experimentos

En el directorio `Pytorch/` se encuentran las implementaciones de los benchmarks, así como del modelo de red neuronal a usar; además de scripts con los experimentos realizados durante el proyecto. Para poder ejecutar cualquier benchmark, es necesario poner los archivos `data.pkl` y `labels.pkl` en el subdirectorio `Datos/`. En caso de estar utilizando datasets con un número de clases diferente a 9, es necesario modificar el  código de los benchmarks, especificando el número de clases a utilizar al modelo. Al mismo tiempo, en caso de querer modificar cualquiera de los hiperparámetros también será necesario modificar el código del programa del benchmark.

Se recomienda usar las scripts de los experimentos para ejecutar los benchmarks.

Ejemplo de uso:

```bash
./lanza_experimentos.sh 2> log &
```

En caso de no querer ejecutar las scripts de experimentos, se debe seguir el siguiente procedimiento:

```bash
python3 benchamrk.py 0 #Nombre del archivo de salida, debe ser un número
python3 agrupa_resultados.py 1 #Número de archivos de resultados a procesar
```

Tras ejecutar los benchmarks, se obtendrá una carpeta con varios archivos `.csv` y uno `.json`, que recogen tanto los parámetros usados para la ejecución, como los resultados obtenidos.
