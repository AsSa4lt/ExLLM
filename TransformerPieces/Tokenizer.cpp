//
// Created by Rostyslav Liapkin on 23.02.2025.
//

#include "Tokenizer.h"
#include <sstream>
#include <fstream>
#include <iostream>

/**
 * Reads vocabulary from txt file
 * @param vocab_file File to read
 */
Tokenizer::Tokenizer(const std::string &vocab_file) {
    std::ifstream file(vocab_file);
    if (!file.is_open()) {
        std::cerr << "Could not open file " << vocab_file << std::endl;
    }
    std::string word;
    int index = 0;
    while (file >> word) {
        if(!word.empty()) {
            vocabulary[word] = index;
            reverse_vocabulary[index] = word;
            index++;
        }
    }
    file.close();
}

/**
 * we are going to tokenize
 * @param input string to tokenize
 * @return vectpr of tokens
 */
std::vector<int> Tokenizer::tokenize(const std::string &text) {
    std::vector<int> tokens;
    std::stringstream ss(text);
    std::string word;
    while (ss >> word) {
        // if we can find the word, than get number of the token
        if(vocabulary.find(word) != vocabulary.end()) {
            tokens.push_back(vocabulary[word]);
        }else {
            tokens.push_back(vocabulary.size());
        }
    }
}

/**
 * Detokenize tokens to human readable text
 * @param tokens tokens
 * @return end results, text
 */
std::string Tokenizer::detokenize(const std::vector<int> &tokens) {
    std::string text;
    for(int token : tokens) {
        // we should look for this token in reverse vocal
        if(reverse_vocabulary.find(token) != reverse_vocabulary.end()) {
            text += reverse_vocabulary[token] + " ";
        }else {
            text += "[UNKNOWN] ";
        }
    }
    return text;
}


