import subprocess, time, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CONFIGS = [
    ("Tiny  (h=128, L=2)",  "weights/tiny.bin",   64),
    ("Small (h=256, L=4)",  "weights/small.bin",  64),
    ("Medium (h=512, L=6)", "weights/medium.bin", 64),
]
RUNS = 3

def bench(binary, weights, tokens):
    times = []
    for _ in range(RUNS):
        r = subprocess.run(
            [binary, "--weights", weights, "--max-tokens", str(tokens), "--runs", "1"],
            capture_output=True, text=True
        )
        for line in r.stdout.splitlines():
            if line.startswith("Run 1:"):
                tps = float(line.split()[2])
                times.append(tps)
    return round(sum(times)/len(times)) if times else 0

binary = "./build/silicon_edge"
labels, tps_vals = [], []

for label, wpath, tokens in CONFIGS:
    print(f"Benchmarking {label}...")
    t = bench(binary, wpath, tokens)
    print(f"  {t} tok/s")
    labels.append(label)
    tps_vals.append(t)

# Plot
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#16213e')

colors = ['#4cc9f0', '#4361ee', '#3a0ca3']
bars = ax.bar(labels, tps_vals, color=colors, width=0.5, zorder=3)
ax.bar_label(bars, fmt=lambda v: f"{int(v):,} tok/s", padding=6,
             fontsize=11, color='white', fontweight='bold')

ax.set_ylabel("Tokens / second", fontsize=12, color='#adb5bd')
ax.set_title("Silicon & Edge — CPU Inference Benchmark\nCustom int4 engine vs model size",
             fontsize=13, color='white', pad=16)
ax.set_ylim(0, max(tps_vals) * 1.35)
ax.tick_params(colors='#adb5bd')
ax.spines[['top','right','left']].set_visible(False)
ax.spines['bottom'].set_color('#333')
ax.yaxis.grid(True, color='#2a2a4a', zorder=0)
ax.set_axisbelow(True)

plt.tight_layout()
os.makedirs("benchmarks", exist_ok=True)
plt.savefig("benchmarks/results.png", dpi=150, bbox_inches='tight')
print("\nSaved benchmarks/results.png")
