cmake_minimum_required(VERSION 3.29)
project(ExLLM)

set(CMAKE_CXX_STANDARD 23)

add_executable(ExLLM main.cpp
        TransformerPieces/Tokenizer.cpp
        TransformerPieces/Tokenizer.h
        TransformerPieces/Embedding.h
        Utils/Matrix.h
        TransformerPieces/SelfAttention.h
        TransformerPieces/FeedForward.h
        TransformerPieces/LayerNorm.h
        TransformerPieces/OutputLayer.h)
