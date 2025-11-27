#include <iostream>
#include <iomanip>
#include <algorithm>
#include "data_loader.h"
#include "mlp.h"

int main()
{
    std::cout << "=== Leyendo CIFAR-10 ===" << std::endl;

    // Carga los 5 batches de entrenamiento
    Batch training_data;
    std::string batch_names[] = {
        "data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin",
        "data_batch_4.bin", "data_batch_5.bin"};

    for (const auto &batch_name : batch_names)
    {
        std::string path = "../data/cifar-10-batches-bin/" + batch_name;
        Batch batch = read_cifar_batch(path);
        training_data.images.insert(training_data.images.end(),
                                    batch.images.begin(), batch.images.end());
        std::cout << "Cargado: " << batch_name << " (" << batch.images.size() << " imagenes)" << std::endl;
    }

    Batch test_data = read_cifar_batch("../data/cifar-10-batches-bin/test_batch.bin");

    std::cout << "\nTotal imágenes de entrenamiento: " << training_data.images.size() << std::endl;
    std::cout << "Total imágenes de prueba: " << test_data.images.size() << std::endl;

    // Crea la red neuronal
    MLP network(0.001);

    int epochs = 35;
    int batch_size = 32;

    std::cout << "\n=== ENTRENAMIENTO ===" << std::endl;

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        double total_loss = 0.0;
        int correct = 0;

        // Itera sobre mini-batches
        for (int batch = 0; batch < training_data.images.size(); batch += batch_size)
        {
            int end = std::min(batch + batch_size, (int)training_data.images.size());

            for (int i = batch; i < end; i++)
            {
                const Image &img = training_data.images[i];
                ActivationCache cache;

                // Forward
                auto prediction = network.forward(img.pixels, cache);
                auto target = one_hot_encode(img.label);

                // Calcula loss
                double loss = network.cross_entropy_loss(prediction, target);
                total_loss += loss;

                // Revisa si acertó
                int pred_label = std::max_element(prediction.begin(), prediction.end()) - prediction.begin();
                if (pred_label == img.label)
                    correct++;

                // Backward
                network.backward(target, cache);
                network.update_weights();
            }
        }

        double avg_loss = total_loss / training_data.images.size();
        double accuracy = (double)correct / training_data.images.size() * 100.0;

        std::cout << "Epoch " << epoch + 1 << "/" << epochs
                  << " | Loss: " << std::fixed << std::setprecision(4) << avg_loss
                  << " | Accuracy: " << accuracy << "%" << std::endl;
    }

    // Validación en datos de prueba
    std::cout << "\n=== VALIDACION (Test Set) ===" << std::endl;
    int correct_test = 0;
    double total_loss_test = 0.0;

    for (const Image &img : test_data.images)
    {
        int pred = network.predict(img.pixels);
        if (pred == img.label)
            correct_test++;

        ActivationCache cache;
        auto prediction = network.forward(img.pixels, cache);
        auto target = one_hot_encode(img.label);
        total_loss_test += network.cross_entropy_loss(prediction, target);
    }

    double test_accuracy = (double)correct_test / test_data.images.size() * 100.0;
    double avg_test_loss = total_loss_test / test_data.images.size();

    std::cout << "Test Loss: " << std::fixed << std::setprecision(4) << avg_test_loss << std::endl;
    std::cout << "Test Accuracy: " << test_accuracy << "%"
              << " (" << correct_test << "/" << test_data.images.size() << ")" << std::endl;

    return 0;
}