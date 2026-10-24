"""Accurate placement recount for llama.cpp sched logs (space-safe tensor names).

The first version of the parser used "(\\S+) \\(size\\)" for the node name, which
silently SKIPS names containing spaces -- ggml names gradients "grad for <tensor>".
This version takes the name as everything between "):" and the "( <size>) [backend]"
group, so those nodes are counted.

Usage: python placement-recount.py <log>
"""
import collections
import re
import sys

path = sys.argv[1]
with open(path, "rb") as fb:
    raw = fb.read()
text = raw.decode("utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8", errors="replace")
text = re.sub(r"\x1b\[[0-9;]*m", "", text)
text = re.sub(r"\d+\.\d+\.\d+\.\d+ [DIWE] ", "\n", text)

SPLIT = re.compile(r"## SPLIT #(\d+): (\S+) # (\d+) inputs")
NODE = re.compile(r"node #\s*(\d+) \(\s*([^)]*?)\s*\):\s*(?P<rest>.*)$")
SIZE = re.compile(r"\(\s*([\d.]+[KMG]?)\s*\)\s*\[\s*([^\]]*?)\s*\]")

splits = collections.Counter()
op_backend = collections.Counter()
cur_split = None
n_nodes = 0

for line in text.splitlines():
    m = SPLIT.search(line)
    if m:
        cur_split = m.group(2)
        splits[cur_split] += 1
        continue
    m = NODE.search(line)
    if not m:
        continue
    op = m.group(2).strip()
    sm = SIZE.search(m.group("rest"))
    if not sm:
        continue
    backend = (sm.group(2).split() or ["?"])[0]
    n_nodes += 1
    op_backend[(op, "CPU" if backend.lower().startswith("cpu") else "GPU")] += 1

total_splits = sum(splits.values())
print("=== splits per backend ===")
for k, v in splits.most_common():
    print("  %-10s %6d  (%4.1f%%)" % (k, v, 100.0 * v / max(1, total_splits)))

print("\n=== printed node instances: %d ===" % n_nodes)
cpu_total = sum(v for (op, b), v in op_backend.items() if b == "CPU")
gpu_total = sum(v for (op, b), v in op_backend.items() if b == "GPU")
print("  on CPU: %d (%.1f%%)   on GPU: %d" % (cpu_total, 100.0 * cpu_total / max(1, n_nodes), gpu_total))

print("\n=== every op that runs on CPU at least once ===")
ops_cpu = collections.Counter({op: v for (op, b), v in op_backend.items() if b == "CPU"})
ops_gpu = collections.Counter({op: v for (op, b), v in op_backend.items() if b == "GPU"})
for op, n in ops_cpu.most_common():
    print("  %-16s CPU %6d   (同样算子 GPU %d)" % (op, n, ops_gpu.get(op, 0)))

print("\n=== OUT_PROD 明细 ===")
c, g = ops_cpu.get("OUT_PROD", 0), ops_gpu.get("OUT_PROD", 0)
print("  OUT_PROD: CPU %d / GPU %d  -> %.1f%% 的权重梯度在 CPU 上算" % (c, g, 100.0 * c / max(1, c + g)))

print("\n=== top ops on GPU ===")
for op, n in ops_gpu.most_common(14):
    print("  %-16s %6d" % (op, n))
