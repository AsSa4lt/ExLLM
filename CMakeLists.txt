cmake_minimum_required(VERSION 3.29)
project(ExLLM)

set(CMAKE_CXX_STANDARD 23)

add_executable(ExLLM main.cpp
        TransformerPieces/Tokenizer.cpp
        TransformerPieces/Tokenizer.h
        TransformerPieces/Embedding.cpp
        TransformerPieces/Embedding.h
        Utils/Matrix.cpp
        Utils/Matrix.h)
