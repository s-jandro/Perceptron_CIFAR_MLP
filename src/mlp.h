#ifndef MLP_H
#define MLP_H

#include <vector>
#include <string>

struct ActivationCache
{
    std::vector<double> z1, a1;
    std::vector<double> z2, a2;
    std::vector<double> z3, a3;
    std::vector<double> input;
};

class MLP
{
private:
    std::vector<std::vector<double>> w1, w2, w3;
    std::vector<double> b1, b2, b3;

    std::vector<std::vector<double>> dw1, dw2, dw3;
    std::vector<double> db1, db2, db3;

    double learning_rate;

public:
    MLP(double lr = 0.01);

    std::vector<double> forward(const std::vector<double> &input, ActivationCache &cache);
    void backward(const std::vector<double> &target, const ActivationCache &cache);
    void update_weights();

    std::vector<double> relu(const std::vector<double> &x);
    std::vector<double> relu_derivative(const std::vector<double> &x);
    std::vector<double> softmax(const std::vector<double> &x);
    std::vector<double> dot_product(const std::vector<std::vector<double>> &W, const std::vector<double> &x);
    std::vector<std::vector<double>> outer_product(const std::vector<double> &a, const std::vector<double> &b);

    double cross_entropy_loss(const std::vector<double> &predicted, const std::vector<double> &target);

    int predict(const std::vector<double> &input);
};

#endif