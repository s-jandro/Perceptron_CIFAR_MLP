MLP para Clasificación de CIFAR-10

Autores:

Renato Corrales Peña

Samuel Iman Quispe

Sergio Paucar Cruz

Descripción del Proyecto

Este proyecto implementa una Red Neuronal Multicapa (MLP) desde cero en C++ para clasificar imágenes del dataset CIFAR-10.
Incluye un cargador propio de datos binarios, una implementación manual del MLP y un ejecutable final que entrena y evalúa el modelo.

El objetivo es comprender los fundamentos del aprendizaje profundo sin utilizar frameworks externos como TensorFlow o PyTorch.

Estructura del Proyecto
build/
└── mlp_cifar10.exe        # Ejecutable principal

data/
└── cifar-10-batches-bin/  # Dataset CIFAR-10 en formato binario
    ├── data_batch_1.bin
    ├── data_batch_2.bin
    ├── data_batch_3.bin
    ├── data_batch_4.bin
    ├── data_batch_5.bin
    └── test_batch.bin

src/
├── data_loader.cpp        # Implementación del cargador de datos
├── data_loader.h
├── mlp.cpp                # Implementación del MLP
├── mlp.h
├── main.cpp               # Ejecución: carga, entrenamiento y test
└── Resultado del ejecutable.txt  # Salida generada por mlp_cifar10.exe

Datasets

El proyecto utiliza los archivos binarios originales de CIFAR-10, que contienen:

50 000 imágenes de entrenamiento

10 000 imágenes de prueba

Imágenes de 32×32 píxeles con 3 canales RGB

El cargador interpreta los binarios y los convierte a vectores utilizables por la red neuronal.

Ejecución

Compila el proyecto (si aún no está compilado):

make


o cualquier sistema de compilación que utilices.

Ejecuta el programa:

./build/mlp_cifar10.exe


El resultado completo de la ejecución está incluido en:
src/Resultado del ejecutable.txt

Ahí podrás encontrar la precisión, errores, tiempos de procesamiento y cualquier métrica generada.
