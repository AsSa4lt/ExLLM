//
// Created by Rostyslav Liapkin on 23.02.2025.
//

#ifndef MATRIX_H
#define MATRIX_H

#include "vector"

template <typename T>
class Matrix {
public:
    std::vector<T> data;
    int rows, cols;

    Matrix(const int rows, const int cols) : rows(rows), cols(cols), data(rows * cols, 0) {}

    T& operator()(const int row, const int col) {
        return data[row * cols + col];
    }

    const T& operator()(const int row, const int col) const {
        return data[row * cols + col];
    }

    Matrix operator+(const Matrix &other) const;
    Matrix operator-(const Matrix &other) const;
    Matrix operator*(const Matrix &other) const;
};




#endif //MATRIX_H
