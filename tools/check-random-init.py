"""Verify that a GGUF model really holds random-initialised weights.

Checks, in order:

1. per-tensor statistics - norms must be exactly 1.0, everything else mean ~0 / std ~ init_std,
2. no structure in the token embeddings - pairwise cosine similarity of random rows ~0,
3. the file reproduces from its config+seed (byte comparison against a fresh build),
4. a different seed produces different bytes.

  python tools/check-random-init.py --model assets/minitrain-6l288-4k-random.gguf \
      --config configs/minitrain-6l288-4k.json
"""

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

T_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
GGML_F32 = 0


def read_gguf(path):
    f = open(path, "rb")
    magic = f.read(4)
    if magic != b"GGUF":
        raise SystemExit(f"{path}: not a GGUF file")
    version, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))

    def rstr():
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n).decode("utf-8", "replace")

    def rval(t):
        if t in T_SIZES:
            n = T_SIZES[t]
            raw = f.read(n)
            fmt = {1: "<B", 2: "<H", 4: "<I", 8: "<Q"}[n]
            if t in (1, 3, 5, 11):
                fmt = {1: "<b", 2: "<h", 4: "<i", 8: "<q"}[n]
            if t in (6, 12):
                fmt = "<f" if t == 6 else "<d"
            return struct.unpack(fmt, raw)[0]
        if t == 8:
            return rstr()
        if t == 9:
            et = struct.unpack("<I", f.read(4))[0]
            n = struct.unpack("<Q", f.read(8))[0]
            return [rval(et) for _ in range(n)]
        raise SystemExit(f"bad gguf type {t}")

    meta = {}
    for _ in range(n_kv):
        k = rstr()
        t = struct.unpack("<I", f.read(4))[0]
        meta[k] = rval(t)

    infos = []
    for _ in range(n_tensors):
        name = rstr()
        nd = struct.unpack("<I", f.read(4))[0]
        dims = struct.unpack("<%dQ" % nd, f.read(8 * nd))
        tt = struct.unpack("<I", f.read(4))[0]
        off = struct.unpack("<Q", f.read(8))[0]
        infos.append((name, dims, tt, off))
    # GGUF has no trailing alignment field: it comes from the general.alignment metadata (32 default)
    align = int(meta.get("general.alignment", 32))

    data_start = f.tell()
    data_start = (data_start + align - 1) // align * align

    tensors = {}
    for name, dims, tt, off in infos:
        if tt != GGML_F32:
            raise SystemExit(f"{name}: expected F32, got type {tt}")
        n = 1
        for d in dims:
            n *= d
        f.seek(data_start + off)
        raw = f.read(4 * n)
        tensors[name] = (dims, np.frombuffer(raw, dtype="<f4").astype(np.float64))
    return meta, tensors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", help="also check reproducibility against this config")
    ap.add_argument("--seed", type=int, default=0, help="alternate seed for the negative control")
    args = ap.parse_args()

    meta, tensors = read_gguf(args.model)
    vocab = len(meta.get("tokenizer.ggml.tokens", []))
    n_params = sum(int(np.prod(d)) for d, _ in tensors.values())

    print(f"model   : {args.model}")
    print(f"meta    : arch={meta.get('general.architecture')} vocab={vocab} "
          f"layers={meta.get('llama.block_count')} embd={meta.get('llama.embedding_length')}")
    print(f"tensors : {len(tensors)}  params {n_params / 1e6:.2f} M")
    print(f"expected: first-step loss ~ ln({vocab}) = {np.log(vocab):.2f}")
    print()

    # ---- 1) per-tensor statistics ----------------------------------------
    print("tensor statistics")
    print(f"  {'tensor':26s} {'shape':>12s} {'mean':>10s} {'std':>10s} {'min':>9s} {'max':>9s}")
    ok_stats = True
    for name in sorted(tensors):
        dims, v = tensors[name]
        is_norm = name.endswith("_norm.weight")
        m, s = float(v.mean()), float(v.std())
        flag = ""
        if is_norm:
            if not (np.allclose(v, 1.0)):
                flag = "  <-- norm is not all ones"
                ok_stats = False
        else:
            if abs(m) > 5e-3:
                flag = "  <-- mean not ~0"
                ok_stats = False
            if not (0.5e-2 <= s <= 5e-2):
                flag = "  <-- std not ~0.02"
                ok_stats = False
        if name in ("token_embd.weight", "output.weight") or "blk.0.attn_q" in name or is_norm:
            print(f"  {name:26s} {str(tuple(int(d) for d in dims)):>12s} "
                  f"{m:10.5f} {s:10.5f} {v.min():9.4f} {v.max():9.4f}{flag}")
    print(f"  ... {len(tensors)} tensors checked: {'OK' if ok_stats else 'FAILED'}")

    # ---- 2) no structure in the embeddings -------------------------------
    print()
    print("token embedding structure")
    dims, emb = tensors["token_embd.weight"]
    rows = int(dims[1])
    rng = np.random.default_rng(0)
    idx = rng.choice(rows, size=min(64, rows), replace=False)
    sub = emb.reshape(rows, -1)[idx]
    sub = sub / np.linalg.norm(sub, axis=1, keepdims=True)
    cos = sub @ sub.T
    off_diag = cos[~np.eye(len(idx), dtype=bool)]
    print(f"  mean |cosine| between random embedding rows: {np.abs(off_diag).mean():.4f} "
          f"(random init -> ~0, trained -> clearly larger)")
    ok_cos = np.abs(off_diag).mean() < 0.05

    # ---- 3) reproducibility + 4) seed sensitivity ------------------------
    ok_repro = None
    if args.config:
        print()
        print("reproducibility")
        # keep the scratch dir inside the workspace: platform temp dirs may not be writable
        td = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "_verify")
        os.makedirs(td, exist_ok=True)
        try:
            a = os.path.join(td, "same.gguf")
            b = os.path.join(td, "other.gguf")
            tool = os.path.join(os.path.dirname(os.path.abspath(__file__)), "make-model.py")
            subprocess.run([sys.executable, tool, "--config", args.config, "--out", a],
                           check=True, stdout=subprocess.DEVNULL)
            subprocess.run([sys.executable, tool, "--config", args.config, "--out", b,
                            "--seed", str(args.seed or 999)], check=True, stdout=subprocess.DEVNULL)

            def sha(p):
                h = hashlib.sha256()
                with open(p, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
                return h.hexdigest()

            same = sha(a) == sha(args.model)
            differ = sha(a) != sha(b)
            print(f"  rebuild with the config seed reproduces this file: {same}")
            print(f"  rebuild with seed {args.seed or 999} gives a different file  : {differ}")
            ok_repro = same and differ
        finally:
            for p in (a, b):
                try:
                    os.remove(p)
                except OSError:
                    pass
            try:
                os.rmdir(td)
            except OSError:
                pass

    print()
    ok = ok_stats and ok_cos and (ok_repro is not False)
    print(f"verdict : {'random-initialised weights' if ok else 'NOT random init - investigate'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
