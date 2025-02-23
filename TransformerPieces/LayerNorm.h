//
// Created by Rostyslav Liapkin on 23.02.2025.
//

#ifndef LAYERNORM_H
#define LAYERNORM_H
#include "../Utils/Matrix.h"
#include <cmath>

// Layer Normalization for Transformer Blocks
// Ensures stable activations by normalizing across embedding dimensions.
template <typename T>
class LayerNorm {
public:
    int embedding_dim; // Number of features per token
    Matrix<T> gamma, beta; // Learnable scaling and shifting parameters
    T epsilon; // Small constant to avoid division by zero

    // Constructor initializes gamma to 1 and beta to 0
    LayerNorm(int embedding_dim, T epsilon = T(1e-5)) : embedding_dim(embedding_dim),
        gamma(1, embedding_dim), beta(1, embedding_dim), epsilon(epsilon) {

        for (int j = 0; j < embedding_dim; ++j) {
            gamma(0, j) = T(1.0); // Scale factor initialized to 1
            beta(0, j) = T(0.0);  // Shift factor initialized to 0
        }
    }

    // Apply LayerNorm to each token independently
    Matrix<T> forward(const Matrix<T>& input) {
        Matrix<T> output = input;
        for (int i = 0; i < input.rows; ++i) {
            // Compute mean across embedding dimension
            T mean = T(0.0);
            for (int j = 0; j < input.cols; ++j) {
                mean = mean + input(i, j);
            }
            mean = mean / T(input.cols);

            // Compute variance across embedding dimension
            T variance = T(0.0);
            for (int j = 0; j < input.cols; ++j) {
                T diff = input(i, j) - mean;
                variance = variance + diff * diff;
            }
            variance = variance / T(input.cols);

            // Normalize each feature
            for (int j = 0; j < input.cols; ++j) {
                output(i, j) = (input(i, j) - mean) / T(std::sqrt(variance.to_float() + epsilon.to_float()));
                output(i, j) = output(i, j) * gamma(0, j) + beta(0, j); // Scale and shift
            }
        }
        return output;
    }
};

#endif //LAYERNORM_H
