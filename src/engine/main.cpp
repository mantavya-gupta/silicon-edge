#include "model.hpp"
#include "ops.hpp"
#include "forward.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <sstream>
#include <cmath>
#include <random>

int sample_topp(const std::vector<float>& logits, float temperature, float topp, std::mt19937& rng){
    int n=logits.size();
    std::vector<float> probs(n);
    float mx=*std::max_element(logits.begin(),logits.end()),sum=0;
    for(int i=0;i<n;i++){probs[i]=expf((logits[i]-mx)/temperature);sum+=probs[i];}
    for(int i=0;i<n;i++) probs[i]/=sum;
    std::vector<int> idx(n); std::iota(idx.begin(),idx.end(),0);
    std::sort(idx.begin(),idx.end(),[&](int a,int b){return probs[a]>probs[b];});
    float cumsum=0; int last=0;
    for(int i=0;i<n;i++){cumsum+=probs[idx[i]];last=i;if(cumsum>=topp)break;}
    std::uniform_real_distribution<float> dist(0,cumsum);
    float r=dist(rng),running=0;
    for(int i=0;i<=last;i++){running+=probs[idx[i]];if(running>=r)return idx[i];}
    return idx[0];
}

int main(int argc,char* argv[]){
    std::string weights="weights/model.bin";
    int max_tokens=20; float temperature=0.1f,topp=0.9f;
    std::vector<int> prompt_tokens={1};
    for(int i=1;i<argc;i++){
        if(std::string(argv[i])=="--weights"&&i+1<argc) weights=argv[++i];
        if(std::string(argv[i])=="--max-tokens"&&i+1<argc) max_tokens=std::stoi(argv[++i]);
        if(std::string(argv[i])=="--temperature"&&i+1<argc) temperature=std::stof(argv[++i]);
        if(std::string(argv[i])=="--prompt-tokens"&&i+1<argc){
            prompt_tokens.clear();
            std::stringstream ss(argv[++i]); std::string tok;
            while(std::getline(ss,tok,',')) prompt_tokens.push_back(std::stoi(tok));
        }
    }
    std::cout<<"Loading model...\n";
    LlamaModel model=LlamaModel::load(weights);
    auto& cfg=model.cfg;
    std::cout<<"Config: hidden="<<cfg.hidden_size<<" layers="<<cfg.num_layers
             <<" heads="<<cfg.num_heads<<" kv_heads="<<cfg.num_kv_heads
             <<" vocab="<<cfg.vocab_size<<" rope_theta="<<cfg.rope_theta<<"\n";

    std::mt19937 rng(42);
    int max_seq=prompt_tokens.size()+max_tokens+10;
    std::vector<KVCache> kv_caches(cfg.num_layers, KVCache(cfg.kv_dim(),max_seq));
    std::vector<float> x(cfg.hidden_size), x_out(cfg.hidden_size);
    std::vector<int> all_tokens=prompt_tokens;

    auto t0=std::chrono::high_resolution_clock::now();

    for(int pos=0;pos<(int)(prompt_tokens.size()+max_tokens);pos++){
        int token=(pos<(int)prompt_tokens.size())?prompt_tokens[pos]:all_tokens.back();
        const float* emb=model.embed_tokens.data()+token*cfg.hidden_size;
        std::copy(emb,emb+cfg.hidden_size,x.data());

        // THE FIX: copy x_out -> x after EACH layer so layers are chained
        for(int l=0;l<(int)model.layers.size();l++){
            layer_forward(model.layers[l],x.data(),x_out.data(),kv_caches[l],pos,cfg);
            std::copy(x_out.begin(),x_out.end(),x.begin()); // <-- this was missing inside loop
        }

        if(pos>=(int)prompt_tokens.size()-1){
            std::vector<float> normed(cfg.hidden_size);
            rmsnorm(x.data(),model.norm.data(),normed.data(),cfg.hidden_size);
            std::vector<float> logits(cfg.vocab_size);
            for(int v=0;v<cfg.vocab_size;v++){
                float dot=0; const float* row=model.lm_head.data()+v*cfg.hidden_size;
                for(int d=0;d<cfg.hidden_size;d++) dot+=row[d]*normed[d];
                logits[v]=dot;
            }
            int next=sample_topp(logits,temperature,topp,rng);
            all_tokens.push_back(next);
            if(next==2||next==128001) break;
        }
    }

    auto t1=std::chrono::high_resolution_clock::now();
    double elapsed=std::chrono::duration<double>(t1-t0).count();
    int generated=all_tokens.size()-prompt_tokens.size();
    std::cout<<"output_tokens:";
    for(int i=prompt_tokens.size();i<(int)all_tokens.size();i++)
        std::cout<<all_tokens[i]<<",";
    std::cout<<"\n"<<generated<<" tokens in "<<(int)elapsed<<"s = "
             <<(int)(generated/elapsed)<<" tok/s\n";
    return 0;
}
