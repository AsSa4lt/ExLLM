//
// Created by Rostyslav Liapkin on 23.02.2025.
//

#include "Matrix.h"

template<typename T>
Matrix<T> Matrix<T>::operator+(const Matrix &other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions do not match");
    }
    Matrix result(rows, cols);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] + other.data[i];
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::operator-(const Matrix &other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions do not match");
    }
    Matrix result(rows, cols);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] - other.data[i];
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::operator*(const Matrix &other) const {
    if (cols != other.rows) {
        throw std::invalid_argument("Matrix dimensions do not match");
    }

    Matrix<T> result(rows, other.cols);

    for (int i = 0; i < rows; ++i) {
        for (int k = 0; k < cols; ++k) {
            T temp = data[i * cols + k]; // Load once to avoid multiple accesses
            for (int j = 0; j < other.cols; ++j) {
                result(i, j) += temp * other(k, j); // Accumulate in-place
            }
        }
    }
    return result;
}
