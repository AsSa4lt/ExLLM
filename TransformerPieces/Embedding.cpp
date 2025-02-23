//
// Created by Rostyslav Liapkin on 23.02.2025.
//

#include "Embedding.h"

/**
 * 
 * @param tokens 
 * @return embedded vector
 */
template<typename T>
Matrix<T> Embedding<T>::get_embedding(const std::vector<int> &tokens)  {
    Matrix<T> embedded(tokens.size(), embedding_dim);
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i] < vocab_size) {
            for (int j = 0; j < embedding_dim; ++j) {
                embedded(i, j) = embedding_matrix(tokens[i], j);
            }
        }
    }
    return embedded;
}