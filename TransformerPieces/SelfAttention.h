#ifndef SELFATTENTION_H
#define SELFATTENTION_H

#include "../Utils/Matrix.h"
#include <vector>
#include <cmath>

template <typename T>
class SelfAttention {
public:
    int embedding_dim;
    Matrix<T> Wq, Wk, Wv;

    SelfAttention(int embedding_dim) : embedding_dim(embedding_dim),
        Wq(embedding_dim, embedding_dim), Wk(embedding_dim, embedding_dim), Wv(embedding_dim, embedding_dim) {
        Wq.randomize();
        Wk.randomize();
        Wv.randomize();
    }

    Matrix<T> compute_attention(const Matrix<T>& input, bool apply_mask = false) {
        int seq_len = input.rows;

        // Step 1: Compute Q (Query), K (Key), and V (Value) matrices by applying learned weights.
        Matrix<T> Q = input * Wq;
        Matrix<T> K = input * Wk;
        Matrix<T> V = input * Wv;

        // Step 2: Compute raw attention scores as Q * K^T.
        // This gives us a measure of how much each token should attend to others.
        Matrix<T> scores = Q * K.transpose();


        // Step 3: Scale the scores to prevent extremely large values before softmax.
        // Scaling by sqrt(embedding_dim) helps stabilize gradients during training.
        float scale_factor = std::sqrt(embedding_dim);
        for (int i = 0; i < scores.rows; ++i) {
            for (int j = 0; j < scores.cols; ++j) {
                scores(i, j) = T(scores(i, j).to_float() / scale_factor);
            }
        }

        // Step 4: Apply causal masking if enabled (for autoregressive decoding).
        // This ensures that future tokens are not attended to, preserving the autoregressive property.
        if (apply_mask) {
            scores.apply_causal_mask();
        }

        /// Step 5: Apply softmax row-wise to normalize the attention scores.
        // Softmax converts the raw scores into probabilities summing to 1 for each row.
        for (int i = 0; i < scores.rows; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < scores.cols; ++j) {
                sum += std::exp(scores(i, j).to_float());
            }
            for (int j = 0; j < scores.cols; ++j) {
                scores(i, j) = T(std::exp(scores(i, j).to_float()) / sum);
            }
        }

        // Step 6: Compute final output by multiplying the normalized attention scores with V.
        // This aggregates the information from different tokens based on their attention weights.
        Matrix<T> output = scores * V;
        return output;
    }
};

#endif // SELFATTENTION_H
