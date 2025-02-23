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

    Matrix operator+(const Matrix &other) const;
    Matrix operator-(const Matrix &other) const;
    Matrix operator*(const Matrix &other) const;

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
