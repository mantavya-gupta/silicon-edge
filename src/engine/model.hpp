#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

struct ModelConfig {
    int hidden_size;
    int num_layers;
    int num_heads;
    int vocab_size;
    int head_dim() const { return hidden_size / num_heads; }
};

struct Q4Matrix {
    std::vector<uint8_t> data;
    std::vector<float>   scales;
    int rows, cols;

    void matmul(const float* input, float* output) const {
        for (int r = 0; r < rows; r++) {
            float sum = 0.0f;
            float scale = scales[r];
            const uint8_t* row = data.data() + r * (cols / 2);
            for (int i = 0; i < cols / 2; i++) {
                uint8_t packed = row[i];
                int8_t w0 = (int8_t)((packed & 0x0F) << 4) >> 4;
                int8_t w1 = (int8_t)(packed & 0xF0) >> 4;
                sum += (float)w0 * input[2*i] + (float)w1 * input[2*i+1];
            }
            output[r] = sum * scale;
        }
    }
};

struct KVCache {
    std::vector<float> keys, vals;
    int head_dim, max_pos = 0;
    KVCache(int hd, int max_seq) : head_dim(hd), keys(max_seq*hd), vals(max_seq*hd) {}
    void push(const float* k, const float* v, int pos) {
        std::copy(k, k+head_dim, keys.data()+pos*head_dim);
        std::copy(v, v+head_dim, vals.data()+pos*head_dim);
        max_pos = pos+1;
    }
    const float* key(int pos) const { return keys.data()+pos*head_dim; }
    const float* val(int pos) const { return vals.data()+pos*head_dim; }
};

struct TransformerLayer {
    Q4Matrix q_proj, k_proj, v_proj, o_proj;
    Q4Matrix gate_proj, up_proj, down_proj;
    std::vector<float> input_norm, post_attn_norm;
};

struct LlamaModel {
    ModelConfig cfg;
    std::vector<TransformerLayer> layers;
    std::vector<float> embed_tokens;
    std::vector<float> lm_head;
    std::vector<float> norm;
    static LlamaModel load(const std::string& path);
};
