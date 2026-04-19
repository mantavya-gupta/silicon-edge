import numpy as np
import struct
import os

def quantize_to_int4(tensor):
    max_val = np.abs(tensor).max(axis=-1, keepdims=True)
    scales = max_val / 7.0
    scales = np.where(scales == 0, 1.0, scales)
    quantized = np.clip(np.round(tensor / scales), -8, 7).astype(np.int8)
    even = quantized[:, 0::2] & 0x0F
    odd  = (quantized[:, 1::2] & 0x0F) << 4
    packed = (even | odd).astype(np.uint8)
    return packed, scales.squeeze(-1).astype(np.float32)

def write_q4(f, rows, cols):
    w = np.random.randn(rows, cols).astype(np.float32) * 0.02
    packed, scales = quantize_to_int4(w)
    f.write(struct.pack('II', rows, cols))
    f.write(scales.tobytes())
    f.write(packed.tobytes())

def write_vec(f, data):
    f.write(data.astype(np.float32).tobytes())

configs = [
    ("weights/tiny.bin",   128,  2,  2,  512),
    ("weights/small.bin",  256,  4,  4,  1024),
    ("weights/medium.bin", 512,  6,  8,  2048),
]

os.makedirs("weights", exist_ok=True)

for path, hidden, layers, heads, vocab in configs:
    ff = hidden * 4
    print(f"Generating {path}  hidden={hidden} layers={layers} heads={heads} vocab={vocab}")
    with open(path, 'wb') as f:
        f.write(struct.pack('IIII', hidden, layers, heads, vocab))
        write_vec(f, np.random.randn(vocab * hidden) * 0.02)   # embed
        write_vec(f, np.ones(hidden))                           # final norm
        for _ in range(layers):
            write_vec(f, np.ones(hidden))   # input norm
            write_vec(f, np.ones(hidden))   # post attn norm
            write_q4(f, hidden, hidden)     # q
            write_q4(f, heads, hidden)      # k  (GQA: 1 head)
            write_q4(f, heads, hidden)      # v
            write_q4(f, hidden, hidden)     # o
            write_q4(f, ff,     hidden)     # gate
            write_q4(f, ff,     hidden)     # up
            write_q4(f, hidden, ff)         # down
        write_vec(f, np.random.randn(hidden * vocab) * 0.02)  # lm_head
    print(f"  saved {os.path.getsize(path)//1024} KB")

print("\nDone. Run: make && ./build/silicon_edge --weights weights/tiny.bin")
