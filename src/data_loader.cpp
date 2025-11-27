// ==================== data_loader.cpp ====================
#include "data_loader.h"
#include <fstream>
#include <iostream>

Batch read_cifar_batch(const std::string &filepath)
{
    Batch batch;
    std::ifstream file(filepath, std::ios::binary);

    if (!file.is_open())
    {
        std::cerr << "Error: No se pudo abrir " << filepath << std::endl;
        return batch;
    }

    for (int i = 0; i < 10000; i++)
    {
        Image img;
        unsigned char label_byte;

        file.read(reinterpret_cast<char *>(&label_byte), 1);
        img.label = static_cast<int>(label_byte);

        img.pixels.resize(3072);
        unsigned char pixel;

        for (int j = 0; j < 3072; j++)
        {
            file.read(reinterpret_cast<char *>(&pixel), 1);
            img.pixels[j] = static_cast<double>(pixel) / 255.0;
        }

        batch.images.push_back(img);
    }

    file.close();
    return batch;
}

std::vector<double> one_hot_encode(int label)
{
    std::vector<double> encoded(10, 0.0);
    encoded[label] = 1.0;
    return encoded;
}
