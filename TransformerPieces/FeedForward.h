//
// Created by Rostyslav Liapkin on 23.02.2025.
//

#ifndef FEEDFORWARD_H
#define FEEDFORWARD_H
#include "../Utils/Matrix.h"

// Feedforward Network (FFN) used in Transformer blocks
// It applies two linear transformations with a non-linearity in between.
// Expands the embedding dimension, applies ReLU activation, and projects back to original size.

template <typename T>
class FeedForward {
public:
    int embedding_dim; // Original embedding dimension
    int hidden_dim;    // Expanded hidden dimension (typically 4x the embedding size)
    Matrix<T> W1, W2;  // Weight matrices for linear transformations
    Matrix<T> b1, b2;  // Bias terms for each linear layer

    // Constructor initializes matrices with random values
    FeedForward(int embedding_dim) : embedding_dim(embedding_dim), hidden_dim(4 * embedding_dim),
        W1(embedding_dim, hidden_dim), W2(hidden_dim, embedding_dim),
        b1(1, hidden_dim), b2(1, embedding_dim) {

        W1.randomize(); // Initialize weights randomly
        W2.randomize();
        b1.randomize(); // Bias terms also initialized randomly
        b2.randomize();
    }

    // ReLU activation function (max(0, x))
    // Introduces non-linearity to allow complex function approximation
    Matrix<T> relu(const Matrix<T>& input) {
        Matrix<T> output = input;
        for (int i = 0; i < output.rows; ++i) {
            for (int j = 0; j < output.cols; ++j) {
                float val = output(i, j).to_float();
                output(i, j) = T(std::max(0.0f, val));
            }
        }
        return output;
    }

    // Forward pass through the feedforward network
    // Applies two linear transformations and a ReLU activation
    Matrix<T> forward(const Matrix<bfloat16>& input) {
        // Step 1: First Linear Transformation (Expanding Features)
        // Multiplies input (batch_size x embedding_dim) with W1 (embedding_dim x hidden_dim)
        // Resulting shape: (batch_size x hidden_dim)
        Matrix<T> hidden = input * W1;

        // Step 2: Add Bias `b1` (Broadcasted)
        // Bias `b1` has shape (1 x hidden_dim), so we add it to each row of `hidden`
        for (int i = 0; i < hidden.rows; ++i) {
            for (int j = 0; j < hidden.cols; ++j) {
                hidden(i, j) = hidden(i, j) + b1(0, j);
            }
        }

        // Step 3: Apply ReLU Activation Function
        // This introduces non-linearity, making the model more expressive
        hidden = relu(hidden);

        // Step 4: Second Linear Transformation (Projecting Back to Original Dimension)
        // Multiplies `hidden` (batch_size x hidden_dim) with W2 (hidden_dim x embedding_dim)
        // Resulting shape: (batch_size x embedding_dim)
        Matrix<T> output = hidden * W2;

        // Step 5: Add Bias `b2` (Broadcasted)
        // Bias `b2` has shape (1 x embedding_dim), so we add it to each row of `output`
        for (int i = 0; i < output.rows; ++i) {
            for (int j = 0; j < output.cols; ++j) {
                output(i, j) = output(i, j) + b2(0, j);
            }
        }

        // Step 6: Return Final Transformed Representation
        // Output maintains the original embedding dimension but now contains refined features
        return output;
    }
};

#endif //FEEDFORWARD_H
