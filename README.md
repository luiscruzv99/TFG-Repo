# Léeme - TFG 

## Descripción

Este repositorio contiene el Trabajo Fin de Grado de Luis Cruz Varona, en el que se examina la diferencia de rendimiento de implementaciones de redes neuronales entre Pytorch y Tensorflow en ambientes Multi-GPU. Para ello se encuentran hechos 2 benchmarks, uno en cada framework, que se sirven de un modelo ResNet34 y de los estándares definidos por [MLCommons](https://mlcommons.org/en/) en sus benchmarks de [Training](https://mlcommons.org/en/training-normal-07/).

### - Estructura del repositorio

El repositorio contiene 3 grandes carpetas: 

- __Data Processing__: Carpeta que contiene los scripts que preprocesan el dataset utilizado por los benchmarks, para ahorrar tiempo de ejecución.
- __Pytorch__: Carpeta que contiene las implementaciones del modelo, bucle de entrenamiento y benchmark utilizando el framework de Pytorch.
- __TensorFlow__: Carpeta que contiene las implementaciones del modelo, bucle de entrenamiento y benchmark utilizando el framework de Tensorflow. 

## Requisitos

Para poder ejecutar los benchmarks, es necesario tener instalado el siguiente software:

-  __Python3__ y __Pip__: para instalar ambos en Ubuntu, hacer:

  ``` bash
  sudo apt update
  sudo apt install python3 python3-pip
  ```

- Librerías de __CUDA__ necesarias para la ejecución de ambos frameworks en GPU: en este caso __CUDA-toolkit__ y __libcudnn7__ además de los drivers de nvidia (versión 450):

  ```bash
  # TODO
  ```

- Frameworks de __Pythorch__ y __Tensorflow__: deben ser instalados a través de un gestor de librerías de Python, en este caso, a través de `pip`, aunque en el caso de __Pytorch__, éste puede ser instalado a través de otros gestores de paquetes:

  ```bash
  pip3 install tensorflow
  pip3 install torch torchvision torchaudio
  ```

  