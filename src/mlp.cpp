#include "mlp.h"
#include <cmath>
#include <random>
#include <algorithm>

MLP::MLP(double lr) : learning_rate(lr)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(0.0, 0.01);

    w1.resize(128, std::vector<double>(3072));
    for (auto &row : w1)
    {
        for (auto &val : row)
        {
            val = dis(gen); // relleno con valores aleatorios
        }
    }
    w2.resize(64, std::vector<double>(128));
    for (auto &row : w2)
    {
        for (auto &val : row)
        {
            val = dis(gen); // relleno con valores aleatorios
        }
    }
    w3.resize(10, std::vector<double>(64));
    for (auto &row : w3)
    {
        for (auto &val : row)
        {
            val = dis(gen); // relleno con valores aleatorios
        }
    }

    b1.resize(128, 0.0);
    b2.resize(64, 0.0);
    b3.resize(10, 0.0);

    dw1.resize(128, std::vector<double>(3072, 0.0));
    dw2.resize(64, std::vector<double>(128, 0.0));
    dw3.resize(10, std::vector<double>(64, 0.0));

    db1.resize(128, 0.0);
    db2.resize(64, 0.0);
    db3.resize(10, 0.0);
}

std::vector<double> MLP::dot_product(const std::vector<std::vector<double>> &W,
                                     const std::vector<double> &x)
{
    int rows = W.size();
    std::vector<double> result(rows, 0.0);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < x.size(); j++)
        {
            result[i] += W[i][j] * x[j];
        }
    }
    return result;
}

std::vector<std::vector<double>> MLP::outer_product(const std::vector<double> &a,
                                                    const std::vector<double> &b)
{
    std::vector<std::vector<double>> result(a.size(), std::vector<double>(b.size()));
    for (int i = 0; i < a.size(); i++)
    {
        for (int j = 0; j < b.size(); j++)
        {
            result[i][j] = a[i] * b[j];
        }
    }
    return result;
}

std::vector<double> MLP::relu(const std::vector<double> &x)
{
    std::vector<double> result = x;
    for (auto &val : result)
        if (val < 0)
            val = 0.0;
    return result;
}

std::vector<double> MLP::relu_derivative(const std::vector<double> &x)
{
    std::vector<double> result(x.size());
    for (int i = 0; i < x.size(); i++)
    {
        result[i] = (x[i] > 0) ? 1.0 : 0.0;
    }
    return result;
}

std::vector<double> MLP::softmax(const std::vector<double> &x)
{
    std::vector<double> result(x.size());
    double sum = 0.0;
    double max_val = *std::max_element(x.begin(), x.end());

    for (int i = 0; i < x.size(); i++)
    {
        result[i] = std::exp(x[i] - max_val);
        sum += result[i];
    }

    for (int i = 0; i < x.size(); i++)
    {
        result[i] /= sum;
    }

    return result;
}

std::vector<double> MLP::forward(const std::vector<double> &input, ActivationCache &cache)
{
    cache.input = input;

    cache.z1 = dot_product(w1, input);
    for (int i = 0; i < 128; i++)
        cache.z1[i] += b1[i];
    cache.a1 = relu(cache.z1);

    cache.z2 = dot_product(w2, cache.a1);
    for (int i = 0; i < 64; i++)
        cache.z2[i] += b2[i];
    cache.a2 = relu(cache.z2);

    cache.z3 = dot_product(w3, cache.a2);
    for (int i = 0; i < 10; i++)
        cache.z3[i] += b3[i];
    cache.a3 = softmax(cache.z3);

    return cache.a3;
}

void MLP::backward(const std::vector<double> &target, const ActivationCache &cache)
{
    // Inicializa gradientes en cero
    for (auto &row : dw1)
        std::fill(row.begin(), row.end(), 0.0);
    for (auto &row : dw2)
        std::fill(row.begin(), row.end(), 0.0);
    for (auto &row : dw3)
        std::fill(row.begin(), row.end(), 0.0);
    std::fill(db1.begin(), db1.end(), 0.0);
    std::fill(db2.begin(), db2.end(), 0.0);
    std::fill(db3.begin(), db3.end(), 0.0);

    // Capa 3: softmax + cross entropy
    std::vector<double> dz3(10);
    for (int i = 0; i < 10; i++)
    {
        dz3[i] = cache.a3[i] - target[i];
    }

    auto dw3_outer = outer_product(dz3, cache.a2);
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 64; j++)
        {
            dw3[i][j] = dw3_outer[i][j];
        }
        db3[i] = dz3[i];
    }

    // Backprop a capa 2
    std::vector<double> da2(64, 0.0);
    for (int j = 0; j < 64; j++)
    {
        for (int i = 0; i < 10; i++)
        {
            da2[j] += w3[i][j] * dz3[i];
        }
    }

    auto relu_deriv_a2 = relu_derivative(cache.z2);
    std::vector<double> dz2(64);
    for (int i = 0; i < 64; i++)
    {
        dz2[i] = da2[i] * relu_deriv_a2[i];
    }

    auto dw2_outer = outer_product(dz2, cache.a1);
    for (int i = 0; i < 64; i++)
    {
        for (int j = 0; j < 128; j++)
        {
            dw2[i][j] = dw2_outer[i][j];
        }
        db2[i] = dz2[i];
    }

    // Backprop a capa 1
    std::vector<double> da1(128, 0.0);
    for (int j = 0; j < 128; j++)
    {
        for (int i = 0; i < 64; i++)
        {
            da1[j] += w2[i][j] * dz2[i];
        }
    }

    auto relu_deriv_a1 = relu_derivative(cache.z1);
    std::vector<double> dz1(128);
    for (int i = 0; i < 128; i++)
    {
        dz1[i] = da1[i] * relu_deriv_a1[i];
    }

    auto dw1_outer = outer_product(dz1, cache.input);
    for (int i = 0; i < 128; i++)
    {
        for (int j = 0; j < 3072; j++)
        {
            dw1[i][j] = dw1_outer[i][j];
        }
        db1[i] = dz1[i];
    }
}

void MLP::update_weights()
{
    for (int i = 0; i < 128; i++)
    {
        for (int j = 0; j < 3072; j++)
        {
            w1[i][j] -= learning_rate * dw1[i][j];
        }
        b1[i] -= learning_rate * db1[i];
    }

    for (int i = 0; i < 64; i++)
    {
        for (int j = 0; j < 128; j++)
        {
            w2[i][j] -= learning_rate * dw2[i][j];
        }
        b2[i] -= learning_rate * db2[i];
    }

    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 64; j++)
        {
            w3[i][j] -= learning_rate * dw3[i][j];
        }
        b3[i] -= learning_rate * db3[i];
    }
}

double MLP::cross_entropy_loss(const std::vector<double> &predicted,
                               const std::vector<double> &target)
{
    double loss = 0.0;
    for (int i = 0; i < 10; i++)
    {
        if (target[i] > 0.5)
        {
            loss -= std::log(predicted[i] + 1e-7);
        }
    }
    return loss;
}

int MLP::predict(const std::vector<double> &input)
{
    ActivationCache cache;
    auto output = forward(input, cache);
    return std::max_element(output.begin(), output.end()) - output.begin();
}
