#pragma once
#include "model.hpp"
#include "ops.hpp"
#include <vector>

inline void layer_forward(const TransformerLayer& layer, const float* x_in,
    float* x_out, KVCache& kv, int pos, const ModelConfig& cfg) {
    int N=cfg.hidden_size, D=cfg.head_dim(), H=cfg.num_heads;
    int KVH=cfg.num_kv_heads, KVD=cfg.kv_dim(), GRP=H/KVH;

    std::vector<float> normed(N),q(N),k(KVD),v(KVD),attn_out(N),ffn_buf(N);
    std::vector<float> x(x_in,x_in+N);

    rmsnorm(x.data(),layer.input_norm.data(),normed.data(),N);
    layer.q_proj.matmul(normed.data(),q.data());
    layer.k_proj.matmul(normed.data(),k.data());
    layer.v_proj.matmul(normed.data(),v.data());

    for(int h=0;h<H;h++)   rope(q.data()+h*D,D,pos);
    for(int h=0;h<KVH;h++) rope(k.data()+h*D,D,pos);
    kv.push(k.data(),v.data(),pos);

    std::fill(attn_out.begin(),attn_out.end(),0.0f);
    float scale=1.0f/sqrtf((float)D);
    std::vector<float> scores(pos+1);

    for(int h=0;h<H;h++){
        int kv_h=h/GRP;
        const float* qh=q.data()+h*D;
        for(int t=0;t<=pos;t++){
            float dot=0; const float* kt=kv.key(t)+kv_h*D;
            for(int d=0;d<D;d++) dot+=qh[d]*kt[d];
            scores[t]=dot*scale;
        }
        softmax(scores.data(),pos+1);
        float* outh=attn_out.data()+h*D;
        for(int t=0;t<=pos;t++){
            const float* vt=kv.val(t)+kv_h*D;
            for(int d=0;d<D;d++) outh[d]+=scores[t]*vt[d];
        }
    }

    layer.o_proj.matmul(attn_out.data(),ffn_buf.data());
    add(x.data(),ffn_buf.data(),N);
    rmsnorm(x.data(),layer.post_attn_norm.data(),normed.data(),N);

    int ff=layer.gate_proj.rows;
    std::vector<float> gate(ff),up(ff),down(N);
    layer.gate_proj.matmul(normed.data(),gate.data());
    layer.up_proj.matmul(normed.data(),up.data());
    silu(gate.data(),ff);
    elemul(gate.data(),up.data(),gate.data(),ff);
    layer.down_proj.matmul(gate.data(),down.data());
    add(x.data(),down.data(),N);
    std::copy(x.begin(),x.end(),x_out);
}
