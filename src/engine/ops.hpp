#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

inline void rmsnorm(const float* x, const float* w, float* out, int n) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / n + 1e-5f);
    for (int i = 0; i < n; i++) out[i] = w[i] * x[i] * ss;
}

inline void softmax(float* x, int n) {
    float max_val = *std::max_element(x, x+n);
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

inline void silu(float* x, int n) {
    for (int i = 0; i < n; i++) x[i] = x[i] / (1.0f + expf(-x[i]));
}

inline void rope(float* x, int head_dim, int pos) {
    for (int i = 0; i < head_dim; i += 2) {
        float freq = 1.0f / powf(10000.0f, (float)i / head_dim);
        float c = cosf(pos * freq), s = sinf(pos * freq);
        float x0 = x[i], x1 = x[i+1];
        x[i]   = x0*c - x1*s;
        x[i+1] = x0*s + x1*c;
    }
}

inline void add(float* a, const float* b, int n) {
    for (int i = 0; i < n; i++) a[i] += b[i];
}

inline void elemul(float* a, const float* b, float* out, int n) {
    for (int i = 0; i < n; i++) out[i] = a[i] * b[i];
}
