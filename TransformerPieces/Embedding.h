//
// Created by Rostyslav Liapkin on 23.02.2025.
//

#ifndef EMBEDDING_H
#define EMBEDDING_H
#include "../Utils/Matrix.h"


/**
 * Embedding is a simple lookup table to map token to a corresponding vector
 */
template <typename T>
class Embedding {
public:
    Matrix<T> token_embedding;
    Matrix<T> position_embedding;
    int vocab_size, embedding_dim, max_seq_length;

    Embedding(int vocab_size, int embedding_dim, int max_seq_length)
        : vocab_size(vocab_size), embedding_dim(embedding_dim), max_seq_length(max_seq_length),
          token_embedding(vocab_size, embedding_dim), position_embedding(max_seq_length, embedding_dim) {

        token_embedding.randomize(); // Random initialize token embeddings
        position_embedding.randomize(); // Random initialize position embeddings
    }

    Matrix<T> get_embedding(const std::vector<int>& tokens){
        int seq_length = tokens.size();
        Matrix<bfloat16> embedded(seq_length, embedding_dim);

        for (size_t i = 0; i < seq_length; ++i) {
            for (int j = 0; j < embedding_dim; ++j) {
                embedded(i, j) = token_embedding(tokens[i], j) + position_embedding(i, j);
            }
        }
        return embedded;
    }
};



#endif //EMBEDDING_H
