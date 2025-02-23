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
    Matrix<T> embedding_matrix;
    int vocab_size, embedding_dim;

    Embedding(int vocab_size, int embedding_dim) : vocab_size(vocab_size), embedding_dim(embedding_dim), embedding_matrix(vocab_size, embedding_dim) {
        embedding_matrix.randomize(); // Use the new random function
    }

    Matrix<T> get_embedding(const std::vector<int>& tokens);
};



#endif //EMBEDDING_H
