#include "model.hpp"
#include <fstream>
#include <stdexcept>

static Q4Matrix read_q4(std::ifstream& f){
    Q4Matrix m;
    f.read((char*)&m.rows,4); f.read((char*)&m.cols,4);
    m.scales.resize(m.rows); f.read((char*)m.scales.data(),m.rows*4);
    m.data.resize(m.rows*(m.cols/2)); f.read((char*)m.data.data(),m.data.size());
    return m;
}
static std::vector<float> read_vec(std::ifstream& f,int n){
    std::vector<float> v(n); f.read((char*)v.data(),n*4); return v;
}

LlamaModel LlamaModel::load(const std::string& path){
    std::ifstream f(path,std::ios::binary);
    if(!f) throw std::runtime_error("Cannot open: "+path);
    LlamaModel m; auto& c=m.cfg;
    f.read((char*)&c.hidden_size,4); f.read((char*)&c.num_layers,4);
    f.read((char*)&c.num_heads,4);   f.read((char*)&c.num_kv_heads,4);
    f.read((char*)&c.vocab_size,4);
    m.embed_tokens=read_vec(f,c.vocab_size*c.hidden_size);
    m.norm=read_vec(f,c.hidden_size);
    m.layers.resize(c.num_layers);
    for(auto& l:m.layers){
        l.input_norm=read_vec(f,c.hidden_size);
        l.post_attn_norm=read_vec(f,c.hidden_size);
        l.q_proj=read_q4(f); l.k_proj=read_q4(f);
        l.v_proj=read_q4(f); l.o_proj=read_q4(f);
        l.gate_proj=read_q4(f); l.up_proj=read_q4(f); l.down_proj=read_q4(f);
    }
    m.lm_head=read_vec(f,c.hidden_size*c.vocab_size);
    return m;
}
