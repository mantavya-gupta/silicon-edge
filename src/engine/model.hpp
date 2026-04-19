#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

struct ModelConfig {
    int hidden_size, num_layers, num_heads, num_kv_heads, vocab_size;
    int head_dim() const { return hidden_size / num_heads; }
    int kv_dim()   const { return head_dim() * num_kv_heads; }
};

struct Q4Matrix {
    std::vector<uint8_t> data;
    std::vector<float> scales;
    int rows, cols;
    void matmul(const float* input, float* output) const {
        for (int r = 0; r < rows; r++) {
            float sum = 0.0f, scale = scales[r];
            const uint8_t* row = data.data() + r * (cols / 2);
            for (int i = 0; i < cols / 2; i++) {
                uint8_t p = row[i];
                // Correct sign extension for 4-bit signed integers
                int8_t w0 = (int8_t)(p << 4) >> 4;  // lower nibble
                int8_t w1 = (int8_t)(p) >> 4;        // upper nibble
                sum += (float)w0 * input[2*i] + (float)w1 * input[2*i+1];
            }
            output[r] = sum * scale;
        }
    }
};

struct KVCache {
    std::vector<float> keys, vals;
    int kv_dim;
    KVCache(int kd, int max_seq) : kv_dim(kd), keys(max_seq*kd), vals(max_seq*kd) {}
    void push(const float* k, const float* v, int pos) {
        std::copy(k, k+kv_dim, keys.data()+pos*kv_dim);
        std::copy(v, v+kv_dim, vals.data()+pos*kv_dim);
    }
    const float* key(int pos) const { return keys.data()+pos*kv_dim; }
    const float* val(int pos) const { return vals.data()+pos*kv_dim; }
};

struct TransformerLayer {
    Q4Matrix q_proj, k_proj, v_proj, o_proj;
    Q4Matrix gate_proj, up_proj, down_proj;
    std::vector<float> input_norm, post_attn_norm;
};

struct LlamaModel {
    ModelConfig cfg;
    std::vector<TransformerLayer> layers;
    std::vector<float> embed_tokens, lm_head, norm;
    static LlamaModel load(const std::string& path);
};
