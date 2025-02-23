//
// Created by Rostyslav Liapkin on 23.02.2025.
//

#ifndef TRANSFORMERBLOCK_H
#define TRANSFORMERBLOCK_H
#include "FeedForward.h"
#include "LayerNorm.h"
#include "SelfAttention.h"

template<typename T>
class TransformerBlock {
public:
    SelfAttention<T> self_attention;
    FeedForward<T> feed_forward;
    LayerNorm<T> layer_norm1;
    LayerNorm<T> layer_norm2;

    explicit TransformerBlock(int embedding_dim)
    : self_attention(embedding_dim),
      feed_forward(embedding_dim),
      layer_norm1(embedding_dim),
      layer_norm2(embedding_dim) {}

    // Forward pass through one Transformer block
    // It always has LayerNorm, then Attention, again LayerNorm and attention
    Matrix<T> forward(const Matrix<T>& input) {
        // Apply LayerNorm before Self-Attention
        Matrix<T> norm1 = layer_norm1.forward(input);

        // Compute Self-Attention
        Matrix<T> attention_output = self_attention.compute_attention(norm1, true);

        // Apply Residual Connection
        attention_output = attention_output + input;

        // Apply LayerNorm before FFN
        Matrix<T> norm2 = layer_norm2.forward(attention_output);

        // Compute FFN Output
        Matrix<T> ffn_output = feed_forward.forward(norm2);

        // Apply Residual Connection
        ffn_output = ffn_output + attention_output;

        return ffn_output;
    }
};

#endif //TRANSFORMERBLOCK_H
