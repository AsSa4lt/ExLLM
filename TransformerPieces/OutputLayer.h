//
// Created by Rostyslav Liapkin on 23.02.2025.
//

#ifndef OUTPUTLAYER_H
#define OUTPUTLAYER_H


#include "../Utils/Matrix.h"
#include <cmath>
#include <vector>

template <typename T>
class OutputLayer {
public:
    int embedding_dim;
    int vocab_size;
    Matrix<T> W_output;
    Matrix<T> b_output;

    OutputLayer(int embedding_dim, int vocab_size)
        : embedding_dim(embedding_dim), vocab_size(vocab_size),
          W_output(embedding_dim, vocab_size), b_output(1, vocab_size) {
        W_output.randomize();
        b_output.randomize();
    }

    // Compute logits (raw scores before softmax)
    Matrix<T> compute_logits(const Matrix<T>& input) {
        // Compute logits (raw scores before softmax)
        Matrix<T> logits = input * W_output; // (batch_size × vocab_size)

        // Broadcast bias across all rows
        for (int i = 0; i < logits.rows; ++i) {
            for (int j = 0; j < logits.cols; ++j) {
                logits(i, j) = logits(i, j) + b_output(0, j); // Copy bias for each row
            }
        }

        return logits; // (batch_size × vocab_size)
    }


    // Softmax function to convert logits into probabilities
    Matrix<T> softmax(const Matrix<T>& logits) {
        Matrix<T> probabilities = logits;
        for (int i = 0; i < probabilities.rows; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < probabilities.cols; ++j) {
                sum += std::exp(probabilities(i, j).to_float());
            }
            for (int j = 0; j < probabilities.cols; ++j) {
                probabilities(i, j) = T(std::exp(probabilities(i, j).to_float()) / sum);
            }
        }
        return probabilities;
    }

    // Select the most probable token (greedy decoding)
    int argmax(const Matrix<T>& probabilities) {
        int max_index = 0;
        float max_value = probabilities(0, 0).to_float();
        for (int j = 1; j < probabilities.cols; ++j) {
            float value = probabilities(0, j).to_float();
            if (value > max_value) {
                max_value = value;
                max_index = j;
            }
        }
        return max_index;
    }
};

#endif //OUTPUTLAYER_H
