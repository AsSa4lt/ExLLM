//
// Created by Rostyslav Liapkin on 23.02.2025.
//

#ifndef TOKENIZER_H
#define TOKENIZER_H
#include <unordered_map>
#include <vector>

/**
 * Tokenizer
 */
class Tokenizer {
public:
    std::unordered_map<std::string, int> vocabulary;
    std::unordered_map<int, std::string> reverse_vocabulary;

    explicit Tokenizer(const std::string& vocab_file);
    std::vector<int> tokenize(const std::string &text);
    std::string detokenize(const std::vector<int> &tokens);
    [[nodiscard]] unsigned long get_vocabulary_size() const { return vocabulary.size(); }
};



#endif //TOKENIZER_H
