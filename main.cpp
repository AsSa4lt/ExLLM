#include <iostream>
#include <vector>
#include <cmath>

#include "TransformerPieces/Embedding.h"
#include "TransformerPieces/FeedForward.h"
#include "TransformerPieces/LayerNorm.h"
#include "TransformerPieces/OutputLayer.h"
#include "TransformerPieces/SelfAttention.h"
#include "TransformerPieces/Tokenizer.h"



int main() {
// 1️⃣ Initialize Tokenizer
    Tokenizer tokenizer = Tokenizer("../vocab.txt");

    // 2️⃣ Create Embedding Layer
    Embedding<bfloat16> embedding(tokenizer.get_vocabulary_size(), 100, 333);

    // 3️⃣ Initialize Transformer Components
    SelfAttention<bfloat16> self_attention(100);
    FeedForward<bfloat16> feed_forward(100);
    LayerNorm<bfloat16> layer_norm1(100);
    LayerNorm<bfloat16> layer_norm2(100);
    OutputLayer<bfloat16> output_layer(100, tokenizer.get_vocabulary_size());

    // 4️⃣ Input Sentence & Tokenization
    std::string input_text = "The cat sat";
    std::vector<int> token_ids = tokenizer.tokenize(input_text);
    Matrix<bfloat16> embedded_tokens = embedding.get_embedding(token_ids);

    // 5️⃣ Iterative Token Generation Loop
    for (int step = 0; step < 10; ++step) {  // Generate 10 tokens
        // Step 1: Apply LayerNorm before Self-Attention
        Matrix<bfloat16> norm1 = layer_norm1.forward(embedded_tokens);

        // Step 2: Compute Self-Attention Output
        Matrix<bfloat16> attention_output = self_attention.compute_attention(norm1, true);

        // Step 3: Apply Residual Connection
        attention_output = attention_output + embedded_tokens;

        // Step 4: Apply LayerNorm before FFN
        Matrix<bfloat16> norm2 = layer_norm2.forward(attention_output);

        // Step 5: Compute FFN Output
        Matrix<bfloat16> ffn_output = feed_forward.forward(norm2);

        // Step 6: Apply Residual Connection
        ffn_output = ffn_output + attention_output;

        // Step 7: Compute Logits (Raw Scores for Each Token)
        Matrix<bfloat16> logits = output_layer.compute_logits(ffn_output);

        // Step 8: Convert Logits to Probabilities (Softmax)
        Matrix<bfloat16> probabilities = output_layer.softmax(logits);

        // Step 9: Select Most Likely Token (Greedy Decoding)
        int next_token = output_layer.argmax(probabilities);
        token_ids.push_back(next_token);

        // Step 10: Convert Token Back to Embedding for Next Step
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
