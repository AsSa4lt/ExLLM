//
// Created by Rostyslav Liapkin on 23.02.2025.
//

#ifndef MATRIX_H
#define MATRIX_H

#include "vector"
#include <random>

struct bfloat16 {
    uint16_t value;

    bfloat16() : value(0) {}
    explicit bfloat16(float f) {
        uint32_t *fptr = reinterpret_cast<uint32_t*>(&f);
        value = static_cast<uint16_t>(*fptr >> 16);
    }
    float to_float() const {
        uint32_t f = static_cast<uint32_t>(value) << 16;
        return *reinterpret_cast<float*>(&f);
    }

    bfloat16 operator+(const bfloat16& other) const {
        return bfloat16(this->to_float() + other.to_float());
    }

    bfloat16 operator-(const bfloat16& other) const {
        return bfloat16(this->to_float() - other.to_float());
    }

    bfloat16 operator*(const bfloat16& other) const {
        return bfloat16(this->to_float() * other.to_float());
    }


    bfloat16 operator/(const bfloat16& other) const {
        return bfloat16(this->to_float() / other.to_float());
    }

    bfloat16& operator+=(const bfloat16& other) {
        *this = bfloat16(this->to_float() + other.to_float());
        return *this;
    }
};

template <typename T>
class Matrix {
public:
    std::vector<T> data;
    int rows, cols;

    Matrix(int rows, int cols) : rows(rows), cols(cols), data(rows * cols, T(0)) {
        if constexpr (std::is_same<T, bfloat16>::value) {
            for (auto &val : data) {
                val = bfloat16(0.0f);
            }
        }
    }

    T& operator()(const int row, const int col) {
        return data[row * cols + col];
    }

    const T& operator()(const int row, const int col) const {
        return data[row * cols + col];
    }

    Matrix operator+(const Matrix &other) const{
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions do not match");
        }
        Matrix result(rows, cols);
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = data[i] + other.data[i];
        }
        return result;
    }

    Matrix operator-(const Matrix &other) const{
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions do not match");
        }
        Matrix result(rows, cols);
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = data[i] - other.data[i];
        }
        return result;
    }

    Matrix operator*(const Matrix &other) const {
        if (cols != other.rows) {
            throw std::invalid_argument("Matrix dimensions do not match");
        }

        Matrix result(rows, other.cols);

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

    Matrix<T> transpose() const {
        Matrix<T> transposed(cols, rows);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                transposed(j, i) = (*this)(i, j);
            }
        }
        return transposed;
    }

    void apply_causal_mask() {
        for (int i = 0; i < rows; ++i) {
            for (int j = i + 1; j < cols; ++j) {
                (*this)(i, j) = T(-1e9);
            }
        }
    }

    void randomize(float min_val = -0.1f, float max_val = 0.1f) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(min_val, max_val);

        for (auto &val : data) {
            val = T(dist(gen));
        }
    }
};




#endif //MATRIX_H
