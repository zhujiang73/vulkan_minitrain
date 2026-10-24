"""Train a byte-level BPE vocabulary from a plain-text corpus (GPT-2 style).

Deliberately dependency-free (stdlib only) and "basic": the classic byte-pair-encoding
algorithm over word frequencies.

  * start from the 256 byte symbols (GPT-2 byte-to-unicode mapping, so every byte is a
    printable character and merges never need an "unknown" fallback),
  * split the corpus into words (whitespace-prefixed, like GPT-2),
  * count word frequencies once, then run `--vocab-size - 256` merges over the *unique*
    words weighted by their counts,
  * write GPT-2-style `vocab.json` + `merges.txt`.

Output feeds `--vocab-json/--vocab-merges` of the model builder (make-model.py), which
writes a GGUF with this tokenizer and randomly initialised tensors.
"""

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict

# GPT-2 byte-level mapping: bytes -> printable unicode chars
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


BYTE_ENCODER = bytes_to_unicode()


# Pre-tokenization. It MUST match what llama.cpp applies when it loads the GGUF, otherwise the
# merges learned here cannot be used there ("missing pre-tokenizer" / degraded quality).
#
# llama.cpp's LLAMA_VOCAB_PRE_TYPE_GPT2 rule (tokenizer.ggml.pre = "gpt-2"):
#   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)
# ASCII transcription of \p{L} -> [A-Za-z] and \p{N} -> [0-9]:
SPLIT_GPT2 = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)")

# Legacy split used by the first vocabularies (" word" / "word," as one piece).
SPLIT_WORD = re.compile(r"\s?\S+|\s+")

SPLIT = SPLIT_GPT2          # --split word switches back


def words_of(text):
    """Split text into BPE pre-tokens; never drops characters (unmatched runs are kept whole)."""
    out = []
    pos = 0
    for m in SPLIT.finditer(text):
        if m.start() > pos:
            out.append(text[pos:m.start()])
        out.append(m.group(0))
        pos = m.end()
    if pos < len(text):
        out.append(text[pos:])
    return out


def count_words(path, max_bytes=0, chunk=64 << 20):
    """Stream the corpus and count whitespace-prefixed words (never holds the file in memory)."""
    counter = Counter()
    n_bytes = 0
    tail = ""
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            n_bytes += len(buf)
            s = tail + buf.decode("utf-8", errors="replace")

            # keep the trailing (possibly incomplete) word for the next round
            cut = max(s.rfind(" "), s.rfind("\n"), s.rfind("\r"), s.rfind("\t"))
            if cut < 0:
                tail = s[-65536:]
                s = s[:-65536] if len(s) > 65536 else ""
            else:
                tail = s[cut:]
                s = s[:cut]

            for w in words_of(s):
                syms = tuple(BYTE_ENCODER[b] for b in w.encode("utf-8"))
                if syms:
                    counter[syms] += 1

            if max_bytes and n_bytes >= max_bytes:
                break
    if tail:
        syms = tuple(BYTE_ENCODER[b] for b in tail.encode("utf-8"))
        if syms:
            counter[syms] += 1
    return counter, n_bytes


def train_from_file(path, vocab_size, max_bytes=0, verbose=True):
    t0 = time.time()
    counter, n_bytes = count_words(path, max_bytes)
    if verbose:
        print(f"  corpus: {n_bytes / 1048576:.1f} MB, {len(counter):,} unique words  ({time.time() - t0:.1f}s)",
              file=sys.stderr)
    return train_counts(counter, vocab_size, verbose, t0)


def train(text, vocab_size, verbose=True):
    counter = Counter()
    for w in words_of(text):
        syms = tuple(BYTE_ENCODER[b] for b in w.encode("utf-8"))
        if syms:
            counter[syms] += 1
    return train_counts(counter, vocab_size, verbose, time.time())


def train_counts(counter, vocab_size, verbose=True, t0=None):
    t0 = t0 or time.time()
    if vocab_size < 300:
        raise ValueError("vocab_size must be >= 300 (256 byte symbols + specials)")
    n_unique = len(counter)
    if verbose:
        print(f"  unique words: {n_unique:,}", file=sys.stderr)

    # specials first (id 0..2) so that <unk>/<s>/</s> stay stable, bytes start at 3
    specials = ["<unk>", "<s>", "</s>"]
    vocab = {}
    for s in specials:
        vocab[s] = len(vocab)
    for b in range(256):
        vocab[BYTE_ENCODER[b]] = len(vocab)

    merges = []
    n_merges = vocab_size - len(vocab)

    # pair stats over unique words (weighted by count)
    symbols = {w: list(w) for w in counter}          # word tuple -> list of symbols
    counts = dict(counter)
    pair_counts = defaultdict(int)
    where = defaultdict(set)                         # pair -> words containing it

    for w, c in counts.items():
        s = symbols[w]
        for i in range(len(s) - 1):
            p = (s[i], s[i + 1])
            pair_counts[p] += c
            where[p].add(w)

    if verbose:
        print(f"  initial pairs: {len(pair_counts):,}  ({time.time() - t0:.1f}s)", file=sys.stderr)

    for step in range(n_merges):
        if not pair_counts:
            break
        best = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
        a, b = best
        new_sym = a + b
        merges.append((a, b))
        vocab[new_sym] = len(vocab)

        affected = where.pop(best, set())
        for w in affected:
            s = symbols[w]
            c = counts[w]
            # subtract old pair counts, rebuild the word with the merge applied
            out = []
            i = 0
            while i < len(s):
                if i < len(s) - 1 and s[i] == a and s[i + 1] == b:
                    out.append(new_sym)
                    i += 2
                else:
                    out.append(s[i])
                    i += 1
            for i in range(len(s) - 1):
                p = (s[i], s[i + 1])
                pair_counts[p] -= c
                if pair_counts[p] <= 0:
                    del pair_counts[p]
                else:
                    where[p].discard(w)
            for i in range(len(out) - 1):
                p = (out[i], out[i + 1])
                pair_counts[p] += c
                where[p].add(w)
            symbols[w] = out

        if verbose and (step + 1) % 500 == 0:
            print(f"  merge {step + 1}/{n_merges}  pairs={len(pair_counts):,}  ({time.time() - t0:.0f}s)",
                  file=sys.stderr)

    return vocab, merges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="plain-text corpus (utf-8)")
    ap.add_argument("--out-dir", required=True, help="where vocab.json / merges.txt go")
    ap.add_argument("--vocab-size", type=int, default=8000)
    ap.add_argument("--max-bytes", type=int, default=0, help="only read this many bytes (0 = all)")
    ap.add_argument("--split", choices=["gpt2", "word"], default="word",
                    help="pre-tokenization: gpt2 = llama.cpp's gpt-2 rule (default), "
                         "word = legacy whitespace-word rule")
    args = ap.parse_args()

    global SPLIT
    SPLIT = SPLIT_GPT2 if args.split == "gpt2" else SPLIT_WORD

    t0 = time.time()
    vocab, merges = train_from_file(args.corpus, args.vocab_size, args.max_bytes)

    # Control tokens must exist as single tokens: llama.cpp splits them into pieces otherwise, and
    # the demo then cannot find its <|endoftext|> story separator (see docs/RUN.md 7.7.1).
    for special in ("<|endoftext|>",):
        if special not in vocab:
            vocab[special] = len(vocab)
            print(f"  reserved special token {special!r} -> id {vocab[special]}", file=sys.stderr)

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    with open(os.path.join(args.out_dir, "merges.txt"), "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        for a, b in merges:
            f.write(f"{a} {b}\n")

    print(f"  done: vocab {len(vocab):,} tokens, {len(merges):,} merges, {time.time() - t0:.1f}s total",
          file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
