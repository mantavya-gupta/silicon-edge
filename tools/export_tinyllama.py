
import torch, struct, numpy as np, os
from transformers import AutoModelForCausalLM

def quantize_to_int4(tensor):
    max_val = np.abs(tensor).max(axis=-1, keepdims=True)
    scales = np.where(max_val==0, 1.0, max_val/7.0)
    quantized = np.clip(np.round(tensor/scales), -8, 7).astype(np.int8)
    # Fix: correct nibble packing
    even = (quantized[:, 0::2] & 0x0F).astype(np.uint8)
    odd  = ((quantized[:, 1::2] & 0x0F) << 4).astype(np.uint8)
    packed = (even | odd).astype(np.uint8)
    return packed, scales.squeeze(-1).astype(np.float32)

def write_q4(f, tensor):
    w = tensor.float().numpy()
    rows, cols = w.shape
    packed, scales = quantize_to_int4(w)
    f.write(struct.pack("II", rows, cols))
    f.write(scales.tobytes())
    f.write(packed.tobytes())

def write_vec(f, tensor):
    f.write(tensor.float().numpy().astype(np.float32).tobytes())

print("Loading TinyLlama...")
model = AutoModelForCausalLM.from_pretrained("/content/tinyllama", dtype=torch.float32)
cfg = model.config

os.makedirs("/content/silicon-edge-git/weights", exist_ok=True)
out = "/content/silicon-edge-git/weights/tinyllama.bin"

with open(out, "wb") as f:
    f.write(struct.pack("IIIII", cfg.hidden_size, cfg.num_hidden_layers,
                        cfg.num_attention_heads, cfg.num_key_value_heads, cfg.vocab_size))
    write_vec(f, model.model.embed_tokens.weight.data)
    write_vec(f, model.model.norm.weight.data)
    for i, layer in enumerate(model.model.layers):
        print(f"  Layer {i+1}/{cfg.num_hidden_layers}...")
        write_vec(f, layer.input_layernorm.weight.data)
        write_vec(f, layer.post_attention_layernorm.weight.data)
        write_q4(f, layer.self_attn.q_proj.weight.data)
        write_q4(f, layer.self_attn.k_proj.weight.data)
        write_q4(f, layer.self_attn.v_proj.weight.data)
        write_q4(f, layer.self_attn.o_proj.weight.data)
        write_q4(f, layer.mlp.gate_proj.weight.data)
        write_q4(f, layer.mlp.up_proj.weight.data)
        write_q4(f, layer.mlp.down_proj.weight.data)
    write_vec(f, model.lm_head.weight.data)

# Verify quantization quality
print("\nVerifying quantization...")
layer0_q = model.model.layers[0].self_attn.q_proj.weight.data.float().numpy()
max_val = np.abs(layer0_q).max(axis=-1, keepdims=True)
scales = np.where(max_val==0, 1.0, max_val/7.0)
quantized = np.clip(np.round(layer0_q/scales), -8, 7)
dequantized = quantized * scales
error = np.abs(layer0_q - dequantized).mean()
print(f"Mean quantization error: {error:.6f} (should be < 0.01)")
print(f"Exported {os.path.getsize(out)//1024//1024} MB")
