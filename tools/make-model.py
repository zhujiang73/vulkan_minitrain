"""Build a trainable f32 GGUF model from a config file - no pretrained checkpoint needed.

This is the "create a model from its description" step a training demo needs:

  * architecture (layers/width/heads/FFN/context/rope) comes from a JSON config,
  * the vocabulary comes from a tokenizer trained on the corpus itself
    (`tools/train-tokenizer.py`, byte-level BPE, GPT-2 style),
  * the weights are random (norms = 1.0, everything else N(0, init_std)).

  python tools/make-model.py --config configs/minitrain-6l288-4k.json \
                             --out assets/minitrain-4k-random.gguf

The output is a plain GGUF, so it loads like any other model:

  vulkan-minitrain --model assets/minitrain-4k-random.gguf --file <corpus> --epochs 1 --lr 1e-5

is therefore a genuine from-scratch run. Sanity check: the first-step loss should be close to
ln(vocab_size) (e.g. ln(4096) = 8.32).

GGUF v3 is written directly here (standard library only; numpy is used when present purely for
faster random number generation).
"""

import argparse
import io
import json
import math
import os
import struct
import sys

try:
    import numpy as np
except ImportError:
    np = None

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_ALIGNMENT = 32

T_UINT32, T_INT32, T_FLOAT32, T_BOOL = 4, 5, 6, 7
T_STRING, T_ARRAY = 8, 9

GGML_TYPE_F32 = 0

TOK_NORMAL, TOK_UNKNOWN, TOK_CONTROL = 1, 2, 3


# --- serialisation helpers --------------------------------------------------
def s_str(v):
    b = v.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def s_u32(v):
    return struct.pack("<I", int(v))


def s_i32(v):
    return struct.pack("<i", int(v))


def s_f32(v):
    return struct.pack("<f", float(v))


def s_bool(v):
    return struct.pack("<B", 1 if v else 0)


def kv(key, typ, payload):
    return s_str(key) + s_u32(typ) + payload


def kv_str(key, v):
    return kv(key, T_STRING, s_str(v))


def kv_u32(key, v):
    return kv(key, T_UINT32, s_u32(v))


def kv_f32(key, v):
    return kv(key, T_FLOAT32, s_f32(v))


def kv_bool(key, v):
    return kv(key, T_BOOL, s_bool(v))


def kv_arr_str(key, values):
    return kv(key, T_ARRAY, s_u32(T_STRING) + struct.pack("<Q", len(values)) + b"".join(s_str(v) for v in values))


def kv_arr_i32(key, values):
    return kv(key, T_ARRAY, s_u32(T_INT32) + struct.pack("<Q", len(values)) + b"".join(s_i32(v) for v in values))


def llama_tensors(n_layer, n_embd, n_head, n_head_kv, n_ff, n_vocab):
    """(name, ne0, ne1); ne0 is the input dimension, matching ggml's convention."""
    head_dim = n_embd // n_head
    n_embd_gqa = head_dim * n_head_kv
    out = [("token_embd.weight", n_embd, n_vocab)]
    for i in range(n_layer):
        out += [
            (f"blk.{i}.attn_norm.weight", n_embd, 1),
            (f"blk.{i}.attn_q.weight", n_embd, n_embd),
            (f"blk.{i}.attn_k.weight", n_embd, n_embd_gqa),
            (f"blk.{i}.attn_v.weight", n_embd, n_embd_gqa),
            (f"blk.{i}.attn_output.weight", n_embd, n_embd),
            (f"blk.{i}.ffn_norm.weight", n_embd, 1),
            (f"blk.{i}.ffn_gate.weight", n_embd, n_ff),
            (f"blk.{i}.ffn_up.weight", n_embd, n_ff),
            (f"blk.{i}.ffn_down.weight", n_ff, n_embd),
        ]
    out += [("output_norm.weight", n_embd, 1), ("output.weight", n_embd, n_vocab)]
    return out


def tensor_bytes(name, ne0, ne1, std, rng_np, rng_py):
    n = ne0 * ne1
    if name.endswith("_norm.weight"):
        vals = np.ones(n, dtype=np.float32) if rng_np is not None else [1.0] * n
        return vals.astype("<f4").tobytes() if rng_np is not None else struct.pack("<%df" % n, *vals)
    if rng_np is not None:
        return (rng_np.standard_normal(n) * std).astype("<f4").tobytes()
    return struct.pack("<%df" % n, *[rng_py.gauss(0.0, std) for _ in range(n)])


def align_up(x, a=GGUF_ALIGNMENT):
    return (x + a - 1) // a * a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", help="default: assets/<config-stem>-random.gguf")
    ap.add_argument("--seed", type=int, default=0, help="override the config seed")
    ap.add_argument("--pre", default="none",
                    help="tokenizer.ggml.pre value; empty string omits the key (llama.cpp then uses 'default')")
    args = ap.parse_args()
    if not args.out:
        stem = os.path.splitext(os.path.basename(args.config))[0]
        args.out = os.path.join("assets", stem + "-random.gguf")

    cfg = json.load(open(args.config, encoding="utf-8"))
    arch = cfg.get("arch", "llama")
    if arch != "llama":
        raise SystemExit(f"only the llama architecture is implemented (config asks for '{arch}')")

    n_layer = int(cfg["n_layer"])
    n_embd = int(cfg["n_embd"])
    n_head = int(cfg["n_head"])
    n_head_kv = int(cfg.get("n_head_kv", n_head))
    n_ff = int(cfg["n_ff"])
    n_ctx = int(cfg.get("n_ctx_train", 1024))
    rms_eps = float(cfg.get("rms_eps", 1e-5))
    rope_freq_base = float(cfg.get("rope_freq_base", 10000.0))
    init_std = float(cfg.get("init_std", 0.02))
    seed = args.seed if args.seed else int(cfg.get("seed", 1234))
    name = cfg.get("name", os.path.basename(args.out))

    vocab_dir = cfg["vocab_dir"]
    vocab = json.load(open(os.path.join(vocab_dir, "vocab.json"), encoding="utf-8"))
    merges = [ln.rstrip("\n") for ln in open(os.path.join(vocab_dir, "merges.txt"), encoding="utf-8")
              if ln.strip() and not ln.startswith("#version")]

    tokens = [None] * len(vocab)
    for tok, idx in vocab.items():
        tokens[idx] = tok
    if any(t is None for t in tokens):
        raise SystemExit("vocab.json ids are not dense (expected 0..n-1)")
    n_vocab = len(tokens)

    tok_types = [TOK_NORMAL] * n_vocab
    for i, t in enumerate(tokens):
        if t == "<unk>":
            tok_types[i] = TOK_UNKNOWN
        elif t in ("<s>", "</s>") or (t.startswith("<|") and t.endswith("|>")):
            # <s>/</s> and <|endoftext|>-style markers must be CONTROL, otherwise llama.cpp
            # warns "control-looking token ... was not control-type" on every load
            tok_types[i] = TOK_CONTROL

    def tid(s):
        return max(int(vocab.get(s, 0)), 0)

    tensors = llama_tensors(n_layer, n_embd, n_head, n_head_kv, n_ff, n_vocab)
    n_params = sum(a * b for _, a, b in tensors)

    print(f"  config : {os.path.basename(args.config)}  arch={arch} L{n_layer} d{n_embd} "
          f"h{n_head}/{n_head_kv} ff{n_ff} ctx{n_ctx}")
    print(f"  vocab  : {n_vocab:,} tokens, {len(merges):,} merges  (from {vocab_dir})")
    print(f"  tensors: {len(tensors)}  params {n_params / 1e6:.2f} M  f32 = {n_params * 4 / 1048576:.1f} MiB")

    # ---- metadata ---------------------------------------------------------
    kvs = [
        kv_str("general.architecture", arch),
        kv_str("general.name", name),
        kv_u32("llama.context_length", n_ctx),
        kv_u32("llama.embedding_length", n_embd),
        kv_u32("llama.block_count", n_layer),
        kv_u32("llama.feed_forward_length", n_ff),
        kv_u32("llama.attention.head_count", n_head),
        kv_u32("llama.attention.head_count_kv", n_head_kv),
        kv_f32("llama.attention.layer_norm_rms_epsilon", rms_eps),
        kv_u32("llama.rope.dimension_count", n_embd // n_head),
        kv_f32("llama.rope.freq_base", rope_freq_base),
        kv_u32("llama.vocab_size", n_vocab),
        kv_str("tokenizer.ggml.model", "gpt2"),
        # without this llama.cpp warns "missing pre-tokenizer type, using: 'default'" and
        # reports degraded generation quality; 'gpt-2' is the rule set this BPE was trained with
        # (inserted below, conditionally, from --pre)
        kv_arr_str("tokenizer.ggml.tokens", tokens),
        kv_arr_i32("tokenizer.ggml.token_type", tok_types),
        kv_arr_str("tokenizer.ggml.merges", merges),
        kv_u32("tokenizer.ggml.unknown_token_id", tid("<unk>")),
        kv_u32("tokenizer.ggml.bos_token_id", tid("<s>")),
        kv_u32("tokenizer.ggml.eos_token_id", tid("</s>")),
        kv_bool("tokenizer.ggml.add_bos_token", False),
        kv_bool("tokenizer.ggml.add_eos_token", False),
        kv_u32("general.alignment", GGUF_ALIGNMENT),   # spec: alignment lives in the metadata
    ]

    # 'none' (or an empty value) means: do not write the key at all, so llama.cpp falls back to
    # its 'default' pre-tokenizer (and prints its warning) -- useful for A/B comparisons
    if args.pre and args.pre.lower() != "none":
        kvs.append(kv_str("tokenizer.ggml.pre", args.pre))

    # ---- layout: header + infos, then aligned data; offsets known upfront ----
    header = io.BytesIO()
    header.write(GGUF_MAGIC)
    header.write(s_u32(GGUF_VERSION))
    header.write(struct.pack("<Q", len(tensors)))
    header.write(struct.pack("<Q", len(kvs)))
    for k in kvs:
        header.write(k)

    sizes = [a * b * 4 for _, a, b in tensors]

    # tensor infos with placeholder offsets; the offset field positions are recorded so they can
    # be patched once the data start is known
    off_positions = []
    for (tname, ne0, ne1) in tensors:
        header.write(s_str(tname))
        header.write(s_u32(2))
        header.write(struct.pack("<QQ", ne0, ne1))
        header.write(s_u32(GGML_TYPE_F32))
        off_positions.append(header.tell())
        header.write(struct.pack("<Q", 0))

    blob = bytearray(header.getvalue())
    data_start = align_up(len(blob))
    pos = data_start
    for i, sz in enumerate(sizes):
        # GGUF tensor offsets are relative to the start of the tensor data area, not the file
        blob[off_positions[i]:off_positions[i] + 8] = struct.pack("<Q", pos - data_start)
        pos = align_up(pos + sz)

    # ---- random weights ---------------------------------------------------
    if np is not None:
        rng_np, rng_py = np.random.default_rng(seed), None
    else:
        import random
        rng_np, rng_py = None, random.Random(seed)
        print("  note   : numpy not available, using the slower stdlib RNG")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    tmp = args.out + ".part"
    pos = data_start
    with open(tmp, "wb") as f:
        f.write(bytes(blob))
        f.write(b"\0" * (data_start - len(blob)))
        for (tname, ne0, ne1), sz in zip(tensors, sizes):
            assert f.tell() == pos, "offset drift"
            f.write(tensor_bytes(tname, ne0, ne1, init_std, rng_np, rng_py))
            pos = align_up(pos + sz)
            f.write(b"\0" * (pos - f.tell()))

    os.replace(tmp, args.out)
    print(f"  wrote  : {args.out}  {os.path.getsize(args.out) / 1048576:.1f} MiB")
    print(f"  expect : first-step loss ~ ln({n_vocab}) = {math.log(n_vocab):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
