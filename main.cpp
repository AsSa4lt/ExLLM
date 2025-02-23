#include <iostream>
#include <vector>
#include <cmath>

#include "TransformerPieces/Embedding.h"
#include "TransformerPieces/FeedForward.h"
#include "TransformerPieces/LayerNorm.h"
#include "TransformerPieces/OutputLayer.h"
#include "TransformerPieces/SelfAttention.h"
#include "TransformerPieces/Tokenizer.h"
#include "TransformerPieces/TransformerBlock.h"


int main() {
    // Initialize Tokenizer
    Tokenizer tokenizer = Tokenizer("../vocab.txt");

    // Create Embedding Layer
    Embedding<bfloat16> embedding(tokenizer.get_vocabulary_size(), 1000, 2048);

    // Define Number of Transformer Layers
    int num_layers = 6; // Change this to add more layers
    std::vector<TransformerBlock<bfloat16>> transformer_layers;
    transformer_layers.reserve(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        transformer_layers.emplace_back(1000); // Embedding dimension = 100
    }

    // Initialize Output Layer
    OutputLayer<bfloat16> output_layer(1000, tokenizer.get_vocabulary_size());

    // Input Sentence & Tokenization
    std::string input_text = "The cat sat";
    std::vector<int> token_ids = tokenizer.tokenize(input_text);
    Matrix<bfloat16> embedded_tokens = embedding.get_embedding(token_ids);

    // Iterative Token Generation Loop
    for (int step = 0; step < 10; ++step) {  // Generate 10 tokens
        Matrix<bfloat16> current_input = embedded_tokens;

        //  Pass Through Multiple Transformer Layers
        for (auto& layer : transformer_layers) {
            current_input = layer.forward(current_input);
        }

        // Compute Logits (Raw Scores for Each Token)
        Matrix<bfloat16> logits = output_layer.compute_logits(current_input);

        // Convert Logits to Probabilities (Softmax)
        Matrix<bfloat16> probabilities = output_layer.softmax(logits);

        // Select Most Likely Token (Greedy Decoding)
        int next_token = output_layer.argmax(probabilities);
        token_ids.push_back(next_token);

        // Convert Token Back to Embedding for Next Step
        embedded_tokens = embedding.get_embedding({next_token});

        // Print the generated token
        std::cout << "Generated Token: " << tokenizer.reverse_vocabulary[next_token] << std::endl;

        // Stop if we generate the `[END]` token
        if (tokenizer.reverse_vocabulary[next_token] == "[END]") {
            break;
        }
    }

    return 0;
}
