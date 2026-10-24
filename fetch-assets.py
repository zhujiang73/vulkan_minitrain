"""Fetch the demo corpora into assets/ (no model download: models are built from a config).

Everything lands in assets/ by default. The full TinyStories train file is large (~1.92 GB),
so it is fetched on demand -- with HTTP Range resume, a progress bar and a sha256 check.

Common uses:
  python fetch-assets.py                              # 40 KB WikiText-2 smoke corpus -> assets/train-small.raw
  python fetch-assets.py --corpus tinystories --slice-mb 10
                                                      # in-domain 10 MB TinyStories slice
  python fetch-assets.py --corpus tinystories --source train --slice-mb 40
                                                      # same, from the train split
  python fetch-assets.py --full-train                 # full TinyStories-train.txt -> assets/

Download details:
  * writes <target>.part next to the target (never the system temp dir), then renames atomically;
  * an interrupted download resumes from the .part size using HTTP Range (no restart from zero);
  * a single-line progress bar with speed and ETA is shown on a TTY (redirections get periodic lines);
  * the sha256 of the finished file is verified against a pinned value (--no-verify to skip).

After the corpora are ready:
  python tools/train-tokenizer.py --corpus assets/TinyStories-train.txt --out-dir assets/vocab-4k --vocab-size 4096
  python tools/make-model.py --config configs/minitrain-6l288-4k.json
  python tools/make-slices.py --slice-mb 390          # then train the slices (README sections 3-4)
"""
import argparse
import hashlib
import os
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)                     # workspace root (this dir is <ws>/vulkan-minitrain)

CORPUS_URL = "https://hf-mirror.com/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"

# TinyStories mirrors, probed at run time: whichever answers first wins.
# stories in both files are separated by <|endoftext|>.
TINYSTORIES_MIRRORS = [
    ("hf-mirror", "https://hf-mirror.com/datasets/roneneldan/TinyStories/resolve/main/"),
    ("huggingface", "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"),
]
TINYSTORIES_BASE = [u for _, u in TINYSTORIES_MIRRORS]      # kept for callers that only need URLs
TINYSTORIES_VALID = "TinyStories-valid.txt"
TINYSTORIES_TRAIN = "TinyStories-train.txt"
TINYSTORIES_TRAIN_SIZE = 1_924_281_556
TINYSTORIES_TRAIN_SHA256 = "c5cf5e22ff13614e830afbe61a99fbcbe8bcb7dd72252b989fa1117a368d401f"
TINYSTORIES_VALID_SIZE = 19_447_282
TINYSTORIES_VALID_SHA256 = "94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4"
STORY_SEP = "<|endoftext|>"



def have(path):
    return os.path.exists(path) and os.path.getsize(path) > 0


def human(nbytes):
    mb = nbytes / 1048576
    return "%.1f MB" % mb if mb < 1024 else "%.2f GB" % (mb / 1024)


def _fmt_duration(sec):
    if sec != sec or sec < 0 or sec == float("inf"):    # nan / negative / inf
        return "--:--"
    sec = int(sec)
    if sec >= 3600:
        return "%d:%02d:%02d" % (sec // 3600, (sec % 3600) // 60, sec % 60)
    return "%02d:%02d" % (sec // 60, sec % 60)


def _rate(done, elapsed):
    """MB/s, or None when the sample is too short to mean anything (first frames)."""
    if done <= 0 or elapsed < 0.5:
        return None
    return done / elapsed / 1048576


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


class _Progress:
    """Single-line progress bar on a TTY, periodic lines when redirected."""

    def __init__(self, label, total, tty=None, width=28):
        self.label = label
        self.total = total or 0
        self.width = width
        self.tty = sys.stdout.isatty() if tty is None else tty
        self.t0 = time.time()
        self.last = 0.0
        self.next_plain = 32 << 20         # redirected: report every 32 MB
        self.done = 0

    def update(self, done, force=False):
        self.done = done
        now = time.time()
        if not self.tty:
            if done >= self.next_plain:
                r = _rate(done, now - self.t0)
                print("    %s  %s / %s  (%s)"
                      % (self.label, human(done), human(self.total) if self.total else "?",
                         "%.1f MB/s" % r if r else "--"), flush=True)
                self.next_plain += 32 << 20
            return
        if not force and now - self.last < 0.2:
            return
        self.last = now
        elapsed = max(now - self.t0, 1e-9)
        r = _rate(done, elapsed)
        if self.total:
            frac = min(done / self.total, 1.0)
            filled = int(self.width * frac)
            bar = "#" * filled + "-" * (self.width - filled)
            eta = _fmt_duration((self.total - done) / (r * 1048576)) if r else "--:--"
            line = ("  %-24s [%s] %5.1f%%  %s / %s  %s  ETA %s"
                    % (self.label[:24], bar, frac * 100, human(done), human(self.total),
                       "%6.1f MB/s" % r if r else "     --  ", eta))
        else:
            line = "  %-24s %s  %s" % (self.label[:24], human(done),
                                       "%6.1f MB/s" % r if r else "     --  ")
        sys.stdout.write("\r" + line + " " * 4)
        sys.stdout.flush()

    def finish(self):
        if self.tty:
            sys.stdout.write("\r" + " " * 118 + "\r")
            sys.stdout.flush()


def _probe(url, timeout=15):
    """Quick 4 KB Range request: tells us whether the mirror is reachable at all.

    A mirror whose TLS handshake hangs would otherwise burn the full download timeout
    on every attempt; probing first keeps that cost down to ~8 s.
    """
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "vulkan-minitrain",
                                                   "Range": "bytes=0-4095"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read(4096)
        return True
    except Exception as e:                         # noqa: BLE001 - any failure means "skip"
        print("  probe failed (%s: %s)" % (type(e).__name__, str(e)[:70]), flush=True)
        return False


def _probe_one(url, timeout=10):
    """Return (ok, seconds, note) for one URL: HEAD first, then a 1 KB Range GET."""
    t0 = time.time()
    for method in ("HEAD", "GET"):
        headers = {"User-Agent": "vulkan-minitrain"}
        if method == "GET":
            headers["Range"] = "bytes=0-1023"
        try:
            req = urllib.request.Request(url, headers=headers, method=method)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                if method == "GET":
                    r.read(1024)
                return (r.status in (200, 206)), time.time() - t0, "HTTP %d" % r.status
        except urllib.error.HTTPError as e:
            if method == "HEAD" and e.code in (403, 405, 501):      # HEAD not allowed -> try GET
                continue
            return False, time.time() - t0, "HTTP %d" % e.code
        except Exception as e:                                     # noqa: BLE001
            if method == "HEAD":                                   # some mirrors kill HEAD
                continue
            return False, time.time() - t0, "%s: %s" % (type(e).__name__, str(e)[:50])
    return False, time.time() - t0, "unreachable"


_MIRROR_CACHE = {}          # name -> base URL decided in this run


def choose_mirror(name, mirrors=None, only=None, timeout=10, budget=15.0):
    """Probe the mirrors for `name` IN PARALLEL and return the first/ fastest one that answers.

    Probing sequentially used to cost the sum of both timeouts (e.g. 11 s mirror + 20 s
    blocked host = 31 s); racing them bounds that at the slowest single probe. Whichever
    answers first wins; the decision is cached for the rest of the run.
    """
    if name in _MIRROR_CACHE:
        return _MIRROR_CACHE[name]

    mirrors = mirrors or TINYSTORIES_MIRRORS
    if only:
        mirrors = [m for m in mirrors if m[0] == only] or mirrors

    print("  probing mirrors for %s (parallel):" % name, flush=True)
    results = {}
    lock = threading.Lock()
    first_ok = threading.Event()

    def worker(label, base):
        ok, secs, note = _probe_one(base + name, timeout=timeout)
        with lock:
            results[label] = (ok, secs, note, base)
            print("    %-14s %-5s %5.1fs  %s" % (label, "OK" if ok else "FAIL", secs, note), flush=True)
        if ok:
            first_ok.set()

    threads = [threading.Thread(target=worker, args=(label, base), daemon=True)
               for label, base in mirrors]
    for th in threads:
        th.start()

    deadline = time.time() + budget
    while time.time() < deadline:
        with lock:
            if any(r[0] for r in results.values()):
                break
        if all(not th.is_alive() for th in threads):
            break
        time.sleep(0.1)

    with lock:
        good = sorted([r for r in results.values() if r[0]], key=lambda r: r[1])
    if not good:
        pending = [th for th in threads if th.is_alive()]
        if pending:
            print("    -> nothing answered within %.0fs; giving up on probing" % budget, flush=True)
        else:
            print("    -> no mirror answered (network down, or the files moved)", flush=True)
        return None

    ok, secs, note, base = good[0]
    label = base.split("/")[2]
    print("    -> using %s (%.1fs, %s)" % (label, secs, note), flush=True)
    _MIRROR_CACHE[name] = base
    return base


def _fetch(url, dst, expected_size=None, expected_sha256=None, retries=5, timeout=45,
           verify=True, label=None, preflight=True):
    """Stream url -> dst with HTTP Range resume, a progress bar and an optional sha256 check.

    The partial download lives in <dst>.part (same directory, same volume), so an
    interrupted run resumes from there instead of restarting at byte 0.
    """
    part = dst + ".part"
    label = label or os.path.basename(dst)

    if preflight and not _probe(url):
        # advisory only: the mirror's first byte can take >10 s when cold, so a failed
        # probe is not proof that the mirror is dead -- the real attempt may still work
        print("  %-24s probe slow/failed, trying anyway" % label, flush=True)

    for attempt in range(1, retries + 1):
        # a .part that already has the full size only needs to be renamed
        if expected_size and os.path.exists(part) and os.path.getsize(part) == expected_size:
            os.replace(part, dst)
            print("  %-24s resumed from a complete .part" % label, flush=True)
            break

        have = os.path.getsize(part) if os.path.exists(part) else 0
        headers = {"User-Agent": "vulkan-minitrain"}
        if have:
            headers["Range"] = "bytes=%d-" % have

        t0 = time.time()
        print("  %-24s connecting%s ..." % (label, " (resume at %s)" % human(have) if have else ""),
              flush=True)
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                status = getattr(r, "status", None) or 200
                ctype = r.headers.get("Content-Type", "")
                if "text/html" in ctype:
                    raise IOError("server returned HTML (mirror error page?) instead of the file")

                if have and status == 206:
                    remaining = int(r.headers.get("Content-Length") or 0)
                    total = have + remaining if remaining else (expected_size or 0)
                    mode = "ab"
                    print("  %-24s resuming at %s" % (label, human(have)), flush=True)
                else:
                    if have and status == 200:
                        print("  %-24s server ignored Range, restarting from 0" % label, flush=True)
                    have = 0
                    clen = int(r.headers.get("Content-Length") or 0)
                    total = clen or expected_size or 0
                    mode = "wb"

                prog = _Progress(label, total)
                done = have
                if have:
                    prog.update(done, force=True)
                with open(part, mode) as f:
                    while True:
                        chunk = r.read(1 << 22)          # 4 MB
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        prog.update(done)
                prog.finish()

            size = os.path.getsize(part)
            if expected_size and size != expected_size:
                print("  %-24s size mismatch (%d of %d bytes), keeping .part for resume"
                      % (label, size, expected_size), flush=True)
                time.sleep(2 * attempt)
                continue

            if verify and expected_sha256:
                t = time.time()
                got = _sha256_file(part)
                if got != expected_sha256:
                    print("  %-24s sha256 MISMATCH (got %s...), restarting from 0"
                          % (label, got[:16]), flush=True)
                    os.remove(part)
                    time.sleep(2 * attempt)
                    continue
                print("  %-24s sha256 %s… OK  (%.0fs, %s in %s)"
                      % (label, got[:16], time.time() - t, human(size),
                         _fmt_duration(time.time() - t0)))

            os.replace(part, dst)
            print("  %-24s done  %s  %.1fs" % (label, human(size), time.time() - t0), flush=True)
            return True

        except urllib.error.HTTPError as e:
            if e.code == 416:                     # stale .part longer than the remote file
                print("  attempt %d/%d: HTTP 416 (stale .part), discarding it" % (attempt, retries))
                if os.path.exists(part):
                    os.remove(part)
                continue
            print("  attempt %d/%d failed: HTTP %d %s" % (attempt, retries, e.code, url))
            if e.code not in (408, 429, 500, 502, 503, 504):
                raise
        except KeyboardInterrupt:
            print("\n  interrupted; partial data kept in %s" % part)
            raise
        except Exception as e:                     # noqa: BLE001 - network errors vary
            print("  attempt %d/%d failed: %s (kept .part for resume)" % (attempt, retries, e))
        time.sleep(3 * attempt)

    if os.path.exists(dst):
        return True
    return False


def download(url, dst, **kw):
    return _fetch(url, dst, **kw)


def download_prefix(name, nbytes, retries=4):
    """Range-download the first nbytes of a TinyStories file (in memory, no temp file)."""
    for attempt in range(1, retries + 1):
        for base in TINYSTORIES_BASE:
            try:
                req = urllib.request.Request(base + name, headers={
                    "User-Agent": "vulkan-minitrain", "Range": "bytes=0-%d" % (nbytes - 1)})
                with urllib.request.urlopen(req, timeout=300) as r:
                    return r.read(nbytes)
            except urllib.error.HTTPError as e:
                print("  prefix attempt %d/%d: HTTP %d (%s)" % (attempt, retries, e.code, base))
                if e.code not in (429, 500, 502, 503, 504):
                    continue
            except Exception as e:                 # noqa: BLE001
                print("  prefix attempt %d/%d: %s" % (attempt, retries, e))
        time.sleep(5 * attempt)
    return None


def fetch_wikitext(assets_dir, verify=True):
    """40 KB slice of WikiText-2: the default smoke-test corpus."""
    corpus = os.path.join(assets_dir, "train-small.raw")
    if have(corpus):
        print("  corpus already present: %s" % corpus)
        return corpus
    zpath = os.path.join(assets_dir, "wikitext-2-raw-v1.zip")
    if not have(zpath):
        ok = False
        for _ in range(1):                        # CORPUS_URL is a single mirror
            ok = download(CORPUS_URL, zpath, verify=False, label="wikitext-2-raw-v1.zip")
        if not ok:
            raise SystemExit("failed to download WikiText-2")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(assets_dir)
    raw = os.path.join(assets_dir, "wikitext-2-raw", "wiki.train.raw")
    with open(raw, "rb") as fh:
        blob = fh.read(40_000)
    blob = blob[:blob.rfind(b"\n") + 1]
    with open(corpus, "wb") as fh:
        fh.write(blob)
    print("  corpus sliced from wiki.train.raw -> %.1f KB" % (len(blob) / 1024))
    return corpus


def full_train_path(assets_dir):
    """Return assets_dir/TinyStories-train.txt when it is present and complete, else None.

    Only assets_dir is considered: the script never looks at (or writes to) other
    directories, so what you see in assets/ is exactly what it uses.
    """
    p = os.path.join(assets_dir, TINYSTORIES_TRAIN)
    if have(p) and os.path.getsize(p) == TINYSTORIES_TRAIN_SIZE:
        return p
    return None


def fetch_full_train(assets_dir, verify=True, only=None):
    """Download the complete TinyStories-train.txt (~1.79 GB) into assets/."""
    dst = os.path.join(assets_dir, TINYSTORIES_TRAIN)
    if have(dst) and os.path.getsize(dst) == TINYSTORIES_TRAIN_SIZE:
        print("  already present and size matches: %s" % dst)
        if verify:
            t = time.time()
            got = _sha256_file(dst)
            ok = got == TINYSTORIES_TRAIN_SHA256
            print("  sha256 %s…  (%.0fs)  %s" % (got[:16], time.time() - t, "OK" if ok else "MISMATCH"))
            if not ok:
                raise SystemExit("sha256 mismatch: got %s, expected %s" % (got, TINYSTORIES_TRAIN_SHA256))
        return dst

    os.makedirs(assets_dir, exist_ok=True)
    print("  fetching %s (%s)" % (TINYSTORIES_TRAIN, human(TINYSTORIES_TRAIN_SIZE)))
    base = choose_mirror(TINYSTORIES_TRAIN, only=only)
    if base and download(base + TINYSTORIES_TRAIN, dst,
                         expected_size=TINYSTORIES_TRAIN_SIZE,
                         expected_sha256=TINYSTORIES_TRAIN_SHA256,
                         verify=verify, label=TINYSTORIES_TRAIN, preflight=False):
        return dst
    raise SystemExit("failed to download TinyStories-train.txt (no mirror reachable? retry later; "
                     "the .part file is kept, so the next run resumes)")


def _story_text(assets_dir, slice_mb, source, verify=True, only=None):
    """Return (text, note) holding at least slice_mb of complete stories."""
    budget = int(slice_mb * 1048576)
    if source == "train":
        full = full_train_path(assets_dir)
        if full:
            with open(full, encoding="utf-8", errors="replace") as fh:
                text = fh.read(int(budget * 1.5))
            cut = text.rfind(STORY_SEP)
            if cut > 0:
                text = text[:cut]
            return text, "local %s (%.2f GB)" % (os.path.basename(full), os.path.getsize(full) / 1e9)
        data = download_prefix(TINYSTORIES_TRAIN, int(budget * 1.5))
        if not data:
            raise SystemExit("failed to fetch a TinyStories-train prefix")
        text = data.decode("utf-8", "replace")
        cut = text.rfind(STORY_SEP)
        if cut > 0:
            text = text[:cut]
        return text, "remote prefix (%.1f MB)" % (len(data) / 1048576)

    src = os.path.join(assets_dir, TINYSTORIES_VALID)
    if have(src) and os.path.getsize(src) == TINYSTORIES_VALID_SIZE:
        print("  source already present: %s (%s)" % (os.path.basename(src), human(TINYSTORIES_VALID_SIZE)))
    else:
        base = choose_mirror(TINYSTORIES_VALID, only=only)
        if not base or not download(base + TINYSTORIES_VALID, src,
                                    expected_size=TINYSTORIES_VALID_SIZE,
                                    expected_sha256=TINYSTORIES_VALID_SHA256,
                                    verify=verify, label=TINYSTORIES_VALID, preflight=False):
            raise SystemExit("failed to download TinyStories-valid.txt (no mirror reachable? retry later)")
    with open(src, encoding="utf-8", errors="replace") as fh:
        return fh.read(), "valid split (%s)" % human(TINYSTORIES_VALID_SIZE)


def fetch_tinystories(assets_dir, slice_mb, source="valid", verify=True, only=None):
    """Build an in-domain corpus: deduplicated TinyStories stories up to ~slice_mb MB."""
    name = "tinystories-%s%gmb.raw" % ("" if source == "valid" else "train-", slice_mb)
    corpus = os.path.join(assets_dir, name)
    if have(corpus):
        print("  corpus already present: %s (%.2f MB)" % (corpus, os.path.getsize(corpus) / 1048576))
        return corpus

    text, note = _story_text(assets_dir, slice_mb, source, verify=verify, only=only)
    stories = [s.strip() for s in text.split(STORY_SEP)]

    empty = short = dup = 0
    seen = set()
    kept = []
    total = 0
    budget = int(slice_mb * 1048576)
    for story in stories:
        if not story:
            empty += 1
            continue
        if len(story) < 80:                       # stray fragments, not real stories
            short += 1
            continue
        if story in seen:
            dup += 1
            continue
        seen.add(story)
        kept.append(story)
        total += len(story.encode("utf-8")) + len(STORY_SEP) + 2
        if total >= budget:
            break

    blob = ("\n" + STORY_SEP + "\n").join(kept) + "\n"
    with open(corpus, "wb") as fh:
        fh.write(blob.encode("utf-8"))

    print("  source:  %s, %d stories, %.2f MB" % (note, len(stories), len(text.encode("utf-8")) / 1048576))
    print("  skipped: %d empty, %d too short (<80 chars), %d exact duplicates" % (empty, short, dup))
    print("  kept:    %d stories -> %s (%.2f MB, ~%.2f M token @4.2 bytes/token)"
          % (len(kept), name, len(blob.encode("utf-8")) / 1048576,
             len(blob.encode("utf-8")) / 4.2 / 1e6))
    print("  note:    the TinyStories train split contains ~15% exact duplicates, so this slice is deduplicated")
    return corpus


def main():
    ap = argparse.ArgumentParser(
        description="Fetch demo corpora into assets/. Models are not downloaded: "
                    "build them with tools/make-model.py from a config. "
                    "Downloads resume from <file>.part and are sha256-checked.")
    ap.add_argument("--assets-dir", default=os.path.join(HERE, "assets"))
    ap.add_argument("--corpus", choices=["wikitext", "tinystories"], default="wikitext",
                    help="wikitext (40 KB smoke test) or tinystories (in-domain slice)")
    ap.add_argument("--source", choices=["valid", "train"], default="valid",
                    help="TinyStories file to slice: valid (18.5 MB) or train (1.79 GB)")
    ap.add_argument("--slice-mb", type=float, default=10.0,
                    help="size of the TinyStories slice in MB (default 10)")
    ap.add_argument("--full-train", action="store_true",
                    help="download the complete TinyStories-train.txt (~1.79 GB) into assets/ and exit")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the sha256 check of downloaded files")
    ap.add_argument("--mirror", choices=["hf-mirror", "huggingface"], default=None,
                    help="force one mirror instead of probing both and using whichever answers first")
    args = ap.parse_args()

    assets_dir = args.assets_dir
    os.makedirs(args.assets_dir, exist_ok=True)
    verify = not args.no_verify

    if args.full_train:
        path = fetch_full_train(assets_dir, verify=verify, only=args.mirror)
        print("\nfull TinyStories train file ready: %s (%s)" % (path, human(os.path.getsize(path))))
        print("next: python tools/make-slices.py --slice-mb 390")
        return 0

    if args.corpus == "tinystories":
        corpus = fetch_tinystories(args.assets_dir, args.slice_mb, args.source, verify=verify,
                                   only=args.mirror)
    else:
        corpus = fetch_wikitext(args.assets_dir, verify=verify)

    print("\ncorpus ready in %s" % args.assets_dir)
    print("  %-28s %8.2f MB" % (os.path.basename(corpus), os.path.getsize(corpus) / 1048576))
    print("\nfor a model, build one from a config (no download needed):")
    print("  python tools/make-model.py --config configs/minitrain-6l288-4k.json")
    print("then train:")
    print("  .\\build\\bin\\Release\\vulkan-minitrain.exe --config configs\\minitrain-6l288-4k.json "
          "--file assets\\%s --epochs 1 --lr 1e-6" % os.path.basename(corpus))
    return 0


if __name__ == "__main__":
    sys.exit(main())
