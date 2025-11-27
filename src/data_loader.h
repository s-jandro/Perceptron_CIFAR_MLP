// ==================== data_loader.h ====================
#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <vector>
#include <string>

struct Image
{
    int label;
    std::vector<double> pixels; // 3072 valores (0.0 - 1.0)
};

struct Batch
{
    std::vector<Image> images;
};

// Lee un archivo binario de CIFAR-10
Batch read_cifar_batch(const std::string &filepath);

// Convierte etiqueta (3) a one-hot [0,0,0,1,0,0,0,0,0,0]
std::vector<double> one_hot_encode(int label);

#endif