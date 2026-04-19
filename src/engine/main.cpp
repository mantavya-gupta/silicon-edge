#include "model.hpp"
#include "ops.hpp"
#include "forward.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <cstring>

int main(int argc, char* argv[]) {
    std::string weights = "weights/model.bin";
    int max_tokens = 128;
    int runs = 5;

    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--weights" && i+1 < argc) weights = argv[++i];
        if (std::string(argv[i]) == "--max-tokens" && i+1 < argc) max_tokens = std::stoi(argv[++i]);
        if (std::string(argv[i]) == "--runs" && i+1 < argc) runs = std::stoi(argv[++i]);
    }

    std::cout << "Loading model from " << weights << "...\n";
    LlamaModel model = LlamaModel::load(weights);
    auto& cfg = model.cfg;
    std::cout << "Config: hidden=" << cfg.hidden_size
              << "  layers=" << cfg.num_layers
              << "  heads=" << cfg.num_heads
              << "  vocab=" << cfg.vocab_size << "\n";

    std::vector<double> tps_results;

    for (int run = 0; run < runs; run++) {
        KVCache kv(cfg.head_dim(), max_tokens + 10);
        std::vector<float> x(cfg.hidden_size);
        std::vector<float> x_out(cfg.hidden_size);

        // Start from token 1 (BOS)
        int token = 1;
        auto t0 = std::chrono::high_resolution_clock::now();

        for (int pos = 0; pos < max_tokens; pos++) {
            // Embed
            const float* emb = model.embed_tokens.data() + token * cfg.hidden_size;
            std::copy(emb, emb + cfg.hidden_size, x.data());

            // Forward through layers
            for (auto& layer : model.layers)
                layer_forward(layer, x.data(), x_out.data(), kv, pos, cfg);

            // Final norm + lm_head (argmax)
            std::vector<float> normed(cfg.hidden_size);
            rmsnorm(x_out.data(), model.norm.data(), normed.data(), cfg.hidden_size);

            std::vector<float> logits(cfg.vocab_size);
            for (int v = 0; v < cfg.vocab_size; v++) {
                float dot = 0.0f;
                const float* row = model.lm_head.data() + v * cfg.hidden_size;
                for (int d = 0; d < cfg.hidden_size; d++) dot += row[d] * normed[d];
                logits[v] = dot;
            }
            token = std::max_element(logits.begin(), logits.end()) - logits.begin();
            std::copy(x_out.begin(), x_out.end(), x.begin());
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        double tps = max_tokens / elapsed;
        tps_results.push_back(tps);
        std::cout << "Run " << run+1 << ": " << (int)tps << " tok/s\n";
    }

    double avg = std::accumulate(tps_results.begin(), tps_results.end(), 0.0) / runs;
    double mn  = *std::min_element(tps_results.begin(), tps_results.end());
    double mx  = *std::max_element(tps_results.begin(), tps_results.end());
    std::cout << "\nAvg: " << (int)avg << " tok/s  |  Min: " << (int)mn << "  |  Max: " << (int)mx << "\n";
    std::cout << "tokens_generated:" << max_tokens << "\n";
    return 0;
}
