"""Split a large training corpus into fixed-size parts for the Vulkan trainer.

Why this exists: the training dataset is held in memory as tokens + inputs + labels,
i.e. roughly three copies of the corpus. A 1.9 GB corpus would need ~19 GB of RAM and
gets killed by the OOM killer, while ~390 MB parts peak around 3.4 GB.

Vocabulary training is NOT done on the slices -- it reads the full corpus file directly
(see tools/train-tokenizer.py), so the tokenizer sees the whole distribution first.

Usage:
  python tools/make-slices.py                                  # assets/TinyStories-train.txt -> assets/ts-full-N.raw (390 MB)
  python tools/make-slices.py --slice-mb 200 --out-dir assets
  python tools/make-slices.py --corpus assets/TinyStories-train.txt --prefix part --slice-mb 100
  python tools/make-slices.py --force                          # rewrite slices that already exist

The split is byte-based and cuts anywhere; TinyStories stories are separated by
<|endoftext|>, so at most one story straddles a boundary and the loss impact of a
half story at a slice edge is negligible.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def human(nbytes):
    mb = nbytes / 1048576
    return "%.0f MB" % mb if mb < 1024 else "%.2f GB" % (mb / 1024)


def estimate_tokens(nbytes, bytes_per_token=3.67):
    """Rough token count with this project's 4K byte-level BPE (measured: 1.92 GB -> 522 M)."""
    return nbytes / bytes_per_token / 1e6


def main():
    ap = argparse.ArgumentParser(
        description="Split a corpus into ~N MB parts so training fits in memory.")
    ap.add_argument("--corpus", default=os.path.join(ROOT, "assets", "TinyStories-train.txt"),
                    help="input corpus (default: assets/TinyStories-train.txt)")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "assets"),
                    help="where to write the slices (default: assets/)")
    ap.add_argument("--prefix", default="ts-full",
                    help="slice file name prefix (default ts-full -> ts-full-1.raw …)")
    ap.add_argument("--slice-mb", type=float, default=390.0,
                    help="target size of each slice in MB (default 390)")
    ap.add_argument("--max-slices", type=int, default=0,
                    help="stop after N slices (0 = all)")
    ap.add_argument("--force", action="store_true",
                    help="rewrite slices even if a same-size file already exists")
    ap.add_argument("--bytes-per-token", type=float, default=3.67,
                    help="only used for the token estimate (default 3.67 = 4K vocab)")
    args = ap.parse_args()

    if not os.path.exists(args.corpus):
        print("  corpus not found: %s" % args.corpus)
        print("  fetch it first:  python fetch-assets.py --full-train")
        return 1
    if args.slice_mb <= 0:
        print("  --slice-mb must be > 0")
        return 1

    total = os.path.getsize(args.corpus)
    target = int(args.slice_mb * 1048576)
    print("  corpus : %s (%s, ~%.0f M token @ %.2f bytes/token)"
          % (os.path.basename(args.corpus), human(total), estimate_tokens(total, args.bytes_per_token),
             args.bytes_per_token))
    print("  slicing: ~%g MB parts -> %s/%s-N.raw" % (args.slice_mb, args.out_dir, args.prefix))
    os.makedirs(args.out_dir, exist_ok=True)

    written = skipped = 0
    idx = 0
    with open(args.corpus, "rb") as src:
        while True:
            if args.max_slices and idx >= args.max_slices:
                break
            idx += 1
            dst = os.path.join(args.out_dir, "%s-%d.raw" % (args.prefix, idx))
            data = src.read(target)
            if not data:
                idx -= 1
                break
            if not args.force and os.path.exists(dst) and os.path.getsize(dst) == len(data):
                print("    %-18s exists   %s" % (os.path.basename(dst), human(len(data))))
                skipped += 1
                continue
            with open(dst, "wb") as out:
                out.write(data)
            print("    %-18s written  %s" % (os.path.basename(dst), human(len(data))))
            written += 1

    if idx == 0:
        print("  nothing to do")
        return 1

    total_out = sum(os.path.getsize(os.path.join(args.out_dir, "%s-%d.raw" % (args.prefix, i)))
                    for i in range(1, idx + 1))
    print("  %d slice(s): %d written, %d already present, total %s (~%.0f M token)"
          % (idx, written, skipped, human(total_out), estimate_tokens(total_out, args.bytes_per_token)))
    rel_dir = os.path.relpath(args.out_dir, ROOT)
    exe = r".\build\bin\Release\vulkan-minitrain.exe"
    tail = "--device Vulkan0 --epochs 1 --lr 1e-6 --stride 1024 --decay off --save-every 30"
    print("\ntrain them one command at a time (slice 1 from the config, the rest chained):")
    for i in range(1, idx + 1):
        flag = "--config configs\\minitrain-6l288-4k.json" if i == 1 else "--load-latest"
        print("  %s %s --file %s\\%s-%d.raw %s"
              % (exe, flag, rel_dir, args.prefix, i, tail))
    return 0


if __name__ == "__main__":
    sys.exit(main())
