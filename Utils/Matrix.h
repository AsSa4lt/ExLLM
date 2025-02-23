//
// Created by Rostyslav Liapkin on 23.02.2025.
//

#ifndef MATRIX_H
#define MATRIX_H

#include "vector"

template <typename T> class Matrix {
 public:
    std::vector<std::vector<T>> data;
    int rows, cols;
    Matrix(int rows, int cols) : rows(rows), cols(cols) {
      data = std::vector<std::vector<T>>(rows, std::vector<T>(cols, 0));
    }

    Matrix operator+(const Matrix &other) const {
        if (rows != other.rows || cols != other.cols) {
          throw std::invalid_argument("Matrix dimensions do not match, operation cannot be performed");
        }
        Matrix result(this->rows, this->cols);

    }
};



#endif //MATRIX_H
