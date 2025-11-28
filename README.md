

---

# MLP para Clasificación de CIFAR-10

**Autores:**

* Renato Corrales Peña
* Samuel Iman Quispe
* Sergio Paucar Cruz

## Descripción del Proyecto

Este proyecto desarrolla una **Red Neuronal Multicapa (MLP)** desde cero en C++ para la clasificación de imágenes del dataset **CIFAR-10**.
Incluye un cargador de datos binarios, una implementación manual del MLP y un ejecutable que entrena y evalúa el modelo.

El objetivo principal es comprender y aplicar los fundamentos del aprendizaje profundo sin utilizar librerías avanzadas como TensorFlow o PyTorch.

---

## Estructura del Proyecto

```
build/
└── mlp_cifar10.exe                # Ejecutable principal

data/
└── cifar-10-batches-bin/          # Dataset CIFAR-10 en formato binario
    ├── data_batch_1.bin
    ├── data_batch_2.bin
    ├── data_batch_3.bin
    ├── data_batch_4.bin
    ├── data_batch_5.bin
    └── test_batch.bin

src/
├── data_loader.cpp                # Implementación del cargador de datos
├── data_loader.h
├── mlp.cpp                        # Implementación del MLP
├── mlp.h
├── main.cpp                       # Código principal: carga, entrenamiento y prueba
└── Resultado del ejecutable.txt   # Resultado generado por mlp_cifar10.exe
```

---

## Dataset CIFAR-10

El proyecto utiliza los archivos binarios originales de CIFAR-10, que contienen:

* 50,000 imágenes de entrenamiento
* 10,000 imágenes de prueba
* Imágenes RGB de 32×32 píxeles
* 10 clases distintas

Los archivos son procesados mediante `data_loader.cpp`, que transforma los binarios en vectores aptos para el MLP.

---

## Ejecución

### 1. Compilación

```bash
make
```

o utilizando tu entorno/compilador preferido.

### 2. Ejecución del programa

```bash
./build/mlp_cifar10.exe
```

### 3. Resultado de la ejecución

El resultado completo producido por el ejecutable se encuentra en:

```
src/Resultado del ejecutable.txt
```

Incluye precisión, errores y otros datos importantes del entrenamiento.

---

## Características del MLP

* Implementación totalmente manual en C++
* Forward y backward propagation
* Funciones de activación (ReLU / Sigmoid dependiendo de la configuración)
* Entrenamiento mediante descenso de gradiente
* Clasificación final en 10 categorías

---

