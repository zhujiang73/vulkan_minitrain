// SPDX-License-Identifier: MIT
// Demo: fine-tune a small f32 GGUF language model on a text file, on one chosen
// GPU backend, using the public llama.cpp training API (llama_opt_*).
//
// It prints the device it resolved, the model, the dataset split, and per epoch the
// training/validation loss and accuracy with wall-clock time. That is enough to see
// whether the work really runs on the requested backend: run it with
// GGML_SCHED_DEBUG=2 and a verbose log to get the per-node backend placement.

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "ggml-opt.h"
#include "ggml-backend.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cerrno>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#  define NOMINMAX            // keep std::min/std::max usable next to windows.h
#  include <direct.h>
#  include <windows.h>
#else
#  include <dirent.h>
#  include <sys/stat.h>
#  include <sys/types.h>
#  include <unistd.h>
#endif

struct demo_args {
    std::string model   = "assets/stories15M.gguf";
    // pretraining = start from random weights. llama.cpp can only load a GGUF, so this points at a
    // GGUF whose metadata/tokenizer are real but whose tensors are re-randomised
    // (fetch-assets.py --random-model). Loading the released checkpoint would be a fine-tune.
    bool        pretrain = false;
    // build the model from a config file instead of loading a checkpoint:
    // assets/<config-stem>-random.gguf is generated with tools/make-model.py when missing
    std::string config;
    std::string pretrain_model = "assets/stories15M-random.gguf";
    std::string corpus  = "assets/train-small.raw";
    // empty = pick the first non-CPU device (so a CUDA-only build works without flags)
    std::string device;
    std::string out;                        // explicit file name; empty = auto-name in save_dir
    std::string save_dir = "output";        // where auto-named models go
    bool        no_save  = false;           // skip saving entirely
    bool        load_latest = false;        // --model = newest .gguf in save_dir
    bool        infer     = false;          // inference only: load model, generate, exit
    std::string prompt    = "Once upon a time, there was a little";
    int         n_predict = 128;
    float       temp      = 0.8f;
    float       top_p     = 0.95f;
    int         top_k     = 40;
    int         seed      = 1234;
    // repetition penalty: small TinyStories models loop without it ("the ball and the ball")
    float       rep_penalty = 1.1f;   // 1.0 = disabled
    int         rep_last_n  = 64;     // how many recent tokens to penalise
    int         epochs  = 2;
    int         n_ctx   = 1024;   // 1024 比 512 快 ~17%：每步做更多 token，摊掉每步固定开销
    int         n_batch = 1024;
    int         n_ubatch = 1024;
    int         n_threads = 0;
    float       lr        = 1e-5f;
    float       val_split = 0.05f;
    // sliding-window stride for the training dataset (0 = n_ctx/2, i.e. half-overlapping windows).
    // n_ctx means non-overlapping: half the steps for the same corpus coverage.
    int         stride    = 0;
    // periodic checkpoint while training (0 = only save at the end); minutes
    int         save_every = 0;
    bool        verbose   = false;

    // learning-rate schedule / weight decay (lr_opt fields that upstream only exposes
    // for -wd; the rest are set directly on params.lr after parsing)
    float       lr_min       = -1.0f;    // >0 enables decay; lr floor
    int         decay_epochs = -1;       // epoch from which lr_min applies
    float       lr_scale     = 0.0f;     // lr = lr0 * 0.5^(epoch * lr_scale)
    float       wd           = -1.0f;    // AdamW weight decay (-1 = leave upstream default)

    // step-level schedule: upstream only decays at epoch boundaries, which is unusable for a
    // multi-hour run (there is no protection inside a long epoch). These drive our own
    // get_opt_pars callback instead (warmup + cosine/linear decay, evaluated per optimizer step).
    int         warmup       = -1;       // -1 = auto (1% of steps), 0 = no warmup, N = steps
    std::string decay        = "cosine"; // off | linear | cosine
};

// ---------------------------------------------------------------------------
// step-level learning-rate schedule
//
// llama.cpp's own lr_opt only decays at epoch boundaries, so a single long epoch has no
// learning-rate protection at all (that is how the 40 MB x 3 epochs run degraded). Our own
// get_opt_pars callback is invoked once per optimizer step, which is the right granularity.
// ---------------------------------------------------------------------------
struct lr_step_schedule {
    int64_t total_steps = 1;
    int64_t warmup      = 0;
    int64_t step        = 0;
    float   lr0         = 1e-5f;
    float   lr_min      = 0.0f;
    int     decay       = 1;   // 0 = off, 1 = cosine, 2 = linear
    float   wd          = 0.0f;
};

static float lr_sched_at(const lr_step_schedule & s, int64_t step) {
    if (s.warmup > 0 && step < s.warmup) {
        // linear warmup from lr0/warmup up to lr0
        return s.lr0 * (float) (step + 1) / (float) s.warmup;
    }
    if (s.decay == 0) {
        return s.lr0;
    }
    const int64_t span = std::max<int64_t>(1, s.total_steps - s.warmup);
    const double  t    = std::min<double>(1.0, (double) (step - s.warmup) / (double) span);
    const double  f    = (s.decay == 1)
        ? 0.5 * (1.0 + std::cos(3.14159265358979323846 * t))   // cosine down to lr_min
        : 1.0 - t;                                             // linear down to lr_min
    return (float) (s.lr_min + (s.lr0 - s.lr_min) * f);
}

// called by ggml-opt once per optimizer step
static ggml_opt_optimizer_params lr_sched_pars(void * userdata) {
    auto * s = (lr_step_schedule *) userdata;

    ggml_opt_optimizer_params res = ggml_opt_get_default_optimizer_params(nullptr);
    res.adamw.alpha = lr_sched_at(*s, s->step);
    res.sgd.alpha   = res.adamw.alpha;
    res.adamw.wd    = s->wd;
    res.sgd.wd      = s->wd;

    if (std::getenv("MINITRAIN_LR_TRACE") != nullptr) {
        fprintf(stderr, "[lr] step %lld/%lld  lr %.6g\n",
                (long long) s->step, (long long) s->total_steps, (double) res.adamw.alpha);
    }

    s->step++;

    return res;
}

static void print_usage(const char * argv0) {
    printf("usage: %s [options]\n", argv0);
    printf("  --model <path>    f32 GGUF model      (default assets/stories15M.gguf)\n");
    printf("  --config <path>   build the model from a config file (implies --pretrain)\n");
    printf("                    generates assets/<config-stem>-random.gguf via tools/make-model.py\n");
    printf("  --pretrain        start from random weights instead of the released checkpoint\n");
    printf("                    (uses assets/stories15M-random.gguf; expect first-step loss ~10.37)\n");
    printf("  --file <path>     training text       (default assets/train-small.raw)\n");
    printf("  --device <name>   backend device      (default: first non-CPU device)\n");
    printf("  --epochs <n>      epochs              (default 2)\n");
    printf("  --lr <f>          AdamW learning rate (default 1e-5)\n");
    printf("  --lr-min <f>      decay floor; >0 enables the schedule\n");
    printf("  --warmup <n|N%%>  step-level warmup, steps or percent (default 1%%)\n");
    printf("  --decay <mode>    step-level decay: off|linear|cosine (default cosine)\n");
    printf("  --decay-epochs <n>  [legacy, ignored] epoch-based decay\n");
    printf("  --lr-scale <f>    lr = lr0 * 0.5^(epoch*lr-scale) (default 0 = flat)\n");
    printf("  --wd <f>          AdamW weight decay (default 0)\n");
    printf("  --ctx <n>         context length      (default 1024; batch/ubatch auto-aligned)\n");
    printf("  --batch <n>       logical batch       (default 512)\n");
    printf("  --ubatch <n>      physical batch      (default 512)\n");
    printf("  --val-split <f>   validation fraction (default 0.05)\n");
    printf("  --save-every <min>  write a checkpoint every N minutes while training (0 = off)\n");
    printf("  --stride <n>      dataset window stride (default n_ctx/2; n_ctx = no overlap,\n");
    printf("                    half the steps for the same coverage)\n");
    printf("  --threads <n>     CPU threads         (default 0 = auto)\n");
    printf("  --out <path>      save the fine-tuned model to this file\n");
    printf("  --save-dir <dir>  directory for auto-named models (default output)\n");
    printf("  --no-save         do not write the trained model\n");
    printf("  --load-latest     load newest .gguf from --save-dir instead of --model\n");
    printf("  --infer           inference only: load the model, generate text, do not train\n");
    printf("  --prompt <s>      inference prompt (default \"Once upon a time, there was a little\")\n");
    printf("  --n-predict <n>   tokens to generate in inference mode (default 128)\n");
    printf("  --temp <f>        sampling temperature (default 0.8)\n");
    printf("  --top-p <f>       nucleus sampling p (default 0.95)\n");
    printf("  --top-k <n>       top-k sampling (default 40)\n");
    printf("  --seed <n>        sampling seed (default 1234)\n");
    printf("  --repeat-penalty <f>  repetition penalty (default 1.1, 1.0 = off)\n");
    printf("  --repeat-last-n <n>   recent tokens to penalise (default 64)\n");
    printf("  -v, --verbose     log everything (needed for GGML_SCHED_DEBUG output)\n");
    printf("  -h, --help        this help\n");
}

static bool file_exists(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    return f.good();
}

// ---- output/ helpers: list saved models, pick the newest, build an auto file name ----

static bool ends_with_gguf(const std::string & s) {
    return s.size() > 5 && s.compare(s.size() - 5, 5, ".gguf") == 0;
}

static std::vector<std::string> list_saved_models(const std::string & dir) {
    std::vector<std::string> out;
#ifdef _WIN32
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA((dir + "\\*.gguf").c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) {
        return out;
    }
    do {
        if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
            out.push_back(dir + "\\" + fd.cFileName);
        }
    } while (FindNextFileA(h, &fd));
    FindClose(h);
#else
    DIR * d = opendir(dir.c_str());
    if (d == nullptr) {
        return out;
    }
    while (struct dirent * e = readdir(d)) {
        const std::string name = e->d_name;
        if (ends_with_gguf(name)) {
            out.push_back(dir + "/" + name);
        }
    }
    closedir(d);
#endif
    return out;
}

static long long file_mtime(const std::string & path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        return 0;
    }
    return (long long) st.st_mtime;
}

static std::string newest_saved_model(const std::string & dir) {
    std::string best;
    long long best_t = -1;
    for (const auto & p : list_saved_models(dir)) {
        const long long t = file_mtime(p);
        if (t > best_t) {
            best_t = t;
            best   = p;
        }
    }
    return best;
}

static std::string base_stem(const std::string & path) {
    const size_t slash = path.find_last_of("/\\");
    std::string name = slash == std::string::npos ? path : path.substr(slash + 1);
    const size_t dot = name.find_last_of('.');
    return dot == std::string::npos ? name : name.substr(0, dot);
}

static std::string make_auto_name(const demo_args & args) {
    char stamp[32];
    const time_t now = time(nullptr);
    struct tm tmv;
#ifdef _WIN32
    localtime_s(&tmv, &now);
#else
    localtime_r(&now, &tmv);
#endif
    strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S", &tmv);

    char lr[32];
    snprintf(lr, sizeof(lr), "%g", (double) args.lr);

    // drop any previous "..._ep<N>_lr<...>_<stamp>" suffix so chained runs do not nest
    std::string stem    = base_stem(args.model);
    const std::string corpus_stem = base_stem(args.corpus);

    // The auto name is <family>__<corpus>_ep<N>_lr<lr>_<stamp>.gguf. The double underscore marks the
    // end of the family, so a chained run (--load-latest on our own output) keeps the same family
    // instead of appending one more corpus segment per slice.
    const size_t sep = stem.find("__");
    if (sep != std::string::npos) {
        stem = stem.substr(0, sep);
    } else {
        const size_t ep = stem.find("_ep");                  // legacy names have no separator
        if (ep != std::string::npos) {
            stem = stem.substr(0, ep);
        }
        const std::string tail = "_" + corpus_stem;          // legacy: model_corpus_corpus
        if (stem.size() > tail.size() && stem.compare(stem.size() - tail.size(), tail.size(), tail) == 0) {
            stem = stem.substr(0, stem.size() - tail.size());
        }
    }

    return stem + "__" + corpus_stem + "_ep" + std::to_string(args.epochs) +
           "_lr" + lr + "_" + stamp + ".gguf";
}

static std::string join_path(const std::string & dir, const std::string & name) {
    if (dir.empty()) {
        return name;
    }
    const char last = dir.back();
    if (last == '/' || last == '\\') {
        return dir + name;
    }
    return dir + "/" + name;
}

static bool make_dir(const std::string & dir) {
#ifdef _WIN32
    return _mkdir(dir.c_str()) == 0 || errno == EEXIST;
#else
    return mkdir(dir.c_str(), 0755) == 0 || errno == EEXIST;
#endif
}

// remind the user when output/ grows; each f32 model is ~94 MB
static void report_save_dir(const std::string & dir, const std::string & saved) {
    const std::vector<std::string> files = list_saved_models(dir);
    long long total = 0;
    for (const auto & p : files) {
        std::ifstream f(p, std::ios::binary | std::ios::ate);
        if (f.good()) {
            total += (long long) f.tellg();
        }
    }
    printf("[demo] saved  : %s\n", saved.c_str());
    printf("[demo] output : %zu model(s) in %s/, %.1f MB total\n", files.size(), dir.c_str(), total / 1048576.0);
    if (files.size() > 4) {
        printf("[demo] WARN  : %s/ holds %zu models (%.1f MB). Clean up old ones, e.g.\n"
               "[demo]         del %s\\*.gguf   (or keep only the newest with --load-latest)\n",
               dir.c_str(), files.size(), total / 1048576.0, dir.c_str());
    }
}

static void print_devices() {
    printf("[demo] available devices (pass the name to --device; CPU means no offload):\n");
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        printf("[demo]   %-10s %s%s\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
               ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU ? "  [cpu, no offload]" : "");
    }
}

// ---- periodic checkpoints while training (6-hour runs cannot afford to lose everything) ----
//
// ggml-opt's epoch callback carries no userdata, so the state lives in a file-scope pointer.
// The callback fires once per datapoint; we forward to the progress bar and, when the interval
// has elapsed, write the current weights next to the final model.

struct checkpoint_state {
    const llama_model * model = nullptr;
    std::string   base_path;      // final path without the .gguf extension
    int           every_s = 0;    // 0 = disabled
    int64_t       last_us = 0;
    int           n_saved = 0;
};

static checkpoint_state * g_ckpt = nullptr;

static double g_first_step_loss = -1.0;   // loss after the first optimizer step

static void epoch_callback_with_checkpoints(bool train, ggml_opt_context_t opt_ctx,
        ggml_opt_dataset_t dataset, ggml_opt_result_t result,
        int64_t idata, int64_t ndata, int64_t t_start_us) {
    ggml_opt_epoch_callback_progress_bar(train, opt_ctx, dataset, result, idata, ndata, t_start_us);

    if (train && g_first_step_loss < 0.0 && idata >= 1) {
        double l = 0.0, unc = 0.0;
        ggml_opt_result_loss(result, &l, &unc);
        g_first_step_loss = l;
    }

    if (!train || g_ckpt == nullptr || g_ckpt->every_s <= 0) {
        return;
    }

    const int64_t now = ggml_time_us();
    if (g_ckpt->last_us == 0) {
        g_ckpt->last_us = now;
        return;
    }
    if (now - g_ckpt->last_us < (int64_t) g_ckpt->every_s * 1000000) {
        return;
    }
    g_ckpt->last_us = now;

    char suffix[64];
    snprintf(suffix, sizeof(suffix), "_step%lld.gguf", (long long) idata);
    const std::string path = g_ckpt->base_path + suffix;

    const auto t0 = std::chrono::steady_clock::now();
    llama_model_save_to_file(g_ckpt->model, path.c_str());
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    if (file_exists(path)) {
        g_ckpt->n_saved++;
        printf("\n[demo] checkpoint: %s (step %lld, %.2f s, %d so far)\n",
                path.c_str(), (long long) idata, secs, g_ckpt->n_saved);
        fflush(stdout);
    } else {
        fprintf(stderr, "\n[demo] checkpoint failed: %s\n", path.c_str());
    }
}

// ---- inference mode: tokenize the prompt, decode, sample token by token ----

static int run_inference(llama_context * ctx, llama_model * model, const demo_args & args) {
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<llama_token> prompt_tokens = common_tokenize(ctx, args.prompt, true, true);

    const int n_ctx = llama_n_ctx(ctx);
    if ((int) prompt_tokens.size() >= n_ctx - 4) {
        // keep room for at least a few generated tokens
        const size_t keep = (size_t) std::max(1, n_ctx - 4);
        prompt_tokens.resize(keep);
        fprintf(stderr, "[demo] prompt truncated to %zu token(s) (n_ctx %d)\n", keep, n_ctx);
    }

    const int n_max = std::min(args.n_predict, n_ctx - (int) prompt_tokens.size());

    printf("[demo] infer  : %zu prompt token(s), n_predict %d, n_ctx %d\n",
           prompt_tokens.size(), n_max, n_ctx);
    printf("[demo] sampler: temp %.2f, top_p %.2f, top_k %d, repeat_penalty %.2f (last %d), seed %d\n",
           (double) args.temp, (double) args.top_p, args.top_k,
           (double) args.rep_penalty, args.rep_last_n, args.seed);
    printf("[demo] -------- prompt --------\n%s", args.prompt.c_str());
    fflush(stdout);

    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (args.rep_penalty > 1.0f && args.rep_last_n > 0) {
        // before truncation, as the llama.cpp docs recommend
        llama_sampler_chain_add(smpl, llama_sampler_init_penalties(
                llama_vocab_n_tokens(vocab), args.rep_last_n, args.rep_penalty, 0.0f, 0.0f));
    }
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(args.top_k));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(args.top_p, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(args.temp));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist((uint32_t) args.seed));

    const auto t0 = std::chrono::steady_clock::now();

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), (int32_t) prompt_tokens.size());
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "[demo] llama_decode failed on the prompt\n");
        llama_sampler_free(smpl);
        return 1;
    }

    // Some models (and GGUFs whose special-token metadata is incomplete) don't flag their
    // end-of-text token as EOG, so it would be printed as a literal "<|endoftext|>" and
    // generation would continue into a second story. Pieces that start with '<' are held back
    // until it is clear whether they form a "<|...|>" control token.
    const bool tok_trace = std::getenv("MINITRAIN_TOK_TRACE") != nullptr;
    std::string pending;

    auto flush_pending = [&]() {
        if (!pending.empty()) {
            fwrite(pending.data(), 1, pending.size(), stdout);
            fflush(stdout);
            pending.clear();
        }
    };

    int n_gen = 0;
    for (; n_gen < n_max; ++n_gen) {
        llama_token tok = llama_sampler_sample(smpl, ctx, -1);
        llama_sampler_accept(smpl, tok);
        if (llama_vocab_is_eog(vocab, tok)) {
            if (tok_trace) { fprintf(stderr, "[tok] %d <eog>\n", tok); }
            break;
        }

        char buf[256];
        const int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, /*special =*/ true);
        if (tok_trace) {
            fprintf(stderr, "[tok] %d n=%d '%s'\n", tok, n, n > 0 ? std::string(buf, (size_t) n).c_str() : "");
        }
        if (n > 0) {
            pending.append(buf, (size_t) n);

            if (pending[0] == '<') {
                // still undecided: keep buffering until we can tell a control token from text
                if (pending.size() >= 2 && pending[1] != '|') {
                    flush_pending();
                } else if (pending.size() >= 4 &&
                           pending[pending.size() - 2] == '|' && pending[pending.size() - 1] == '>') {
                    if (tok_trace) { fprintf(stderr, "[tok] control token '%s' -> stop\n", pending.c_str()); }
                    pending.clear();   // never print the control token
                    break;             // "<|...|>": end of this story
                } else if (pending.size() > 40) {
                    flush_pending();   // not a marker after all
                }
            } else {
                flush_pending();
            }
        }

        batch = llama_batch_get_one(&tok, 1);
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "\n[demo] llama_decode failed while generating\n");
            break;
        }
    }
    // If we stopped because --n-predict ran out while a "<|..." prefix was still buffered, that is
    // a half-written control token rather than text: drop it instead of printing "<|endoftext".
    if (pending.size() >= 2 && pending[0] == '<' && pending[1] == '|') {
        pending.clear();
    }
    flush_pending();

    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("\n[demo] -------- end -----------\n");
    printf("[demo] generated %d token(s) in %.2f s (%.1f tok/s)\n",
           n_gen, secs, secs > 0 ? n_gen / secs : 0.0);

    llama_sampler_free(smpl);
    return 0;
}

// Tokenizing a corpus that contains <|endoftext|> markers with parse_special = true makes
// llama.cpp take >10 minutes for 390 MB (it re-scans for special tokens), while the markers only
// matter as story boundaries. Splitting at the marker and tokenizing each story with the ordinary
// fast path -- then inserting the marker's token id -- yields exactly the same token sequence,
// because the marker is a single token in the reference encoding: no BPE merge can span it.
//
// The pieces are independent, so they can be tokenized in parallel. `plan` records the document
// order (piece index, or -1 for a separator), which keeps leading / trailing / consecutive
// separators exact and makes the parallel result identical to the sequential one.
static std::vector<llama_token> tokenize_corpus(
        llama_context * ctx, const std::string & text, const std::string & sep,
        std::atomic<size_t> * progress) {
    std::vector<llama_token> sep_tok = common_tokenize(ctx, sep, false, true);
    if (sep_tok.size() != 1) {                     // unusual vocab: fall back to the slow path
        fprintf(stderr, "[demo] note   : separator '%s' is not a single token, using the slow path\n",
                sep.c_str());
        return common_tokenize(ctx, text, true, true);
    }
    const llama_token sep_id = sep_tok[0];

    struct piece { size_t off, len; };
    std::vector<piece> pieces;
    std::vector<int>   plan;                       // >= 0: piece index, -1: separator

    size_t pos = 0;
    while (true) {
        const size_t hit = text.find(sep, pos);
        if (hit == std::string::npos) {
            if (pos < text.size()) {
                plan.push_back((int) pieces.size());
                pieces.push_back({pos, text.size() - pos});
            }
            break;
        }
        if (hit > pos) {
            plan.push_back((int) pieces.size());
            pieces.push_back({pos, hit - pos});
        }
        plan.push_back(-1);
        pos = hit + sep.size();
    }

    // Single-threaded by default: 390 MB takes ~130 s, and the multi-threaded variant turned out to
    // be unsafe (>=4 threads crash; 2 threads also change the token count), so it is opt-in only.
    unsigned n_threads = 1;
    if (const char * e = std::getenv("MINITRAIN_TOK_THREADS")) {   // experimental
        n_threads = (unsigned) std::max(1, atoi(e));
    }
    if (pieces.size() < (size_t) n_threads * 8) {
        n_threads = 1;
    }

    std::vector<std::vector<llama_token>> part(n_threads);
    std::vector<std::vector<std::pair<size_t, size_t>>> ranges(n_threads);

    auto chunk_of = [&](size_t piece_idx) -> unsigned {
        return (unsigned) std::min<size_t>(n_threads - 1, piece_idx * n_threads / pieces.size());
    };
    auto chunk_begin = [&](unsigned c) -> size_t {
        return pieces.size() * c / n_threads;
    };

    auto tokenize_range = [&](unsigned c) {
        const size_t begin = chunk_begin(c);
        const size_t end   = chunk_begin(c + 1);
        std::vector<llama_token> & dst = part[c];
        auto & rng = ranges[c];
        rng.resize(end - begin);
        size_t bytes = 0;
        for (size_t k = begin; k < end; ++k) {
            const piece & p = pieces[k];
            const size_t b = dst.size();
            std::vector<llama_token> toks =
                common_tokenize(ctx, text.substr(p.off, p.len), false, false);
            dst.insert(dst.end(), toks.begin(), toks.end());
            rng[k - begin] = {b, dst.size()};
            bytes += p.len;
            if (progress && (k % 256) == 0) {
                progress->store(p.off + p.len, std::memory_order_relaxed);
            }
        }
    };

    if (n_threads <= 1) {
        tokenize_range(0);
    } else {
        std::vector<std::thread> th;
        th.reserve(n_threads);
        for (unsigned c = 0; c < n_threads; ++c) {
            th.emplace_back(tokenize_range, c);
        }
        for (auto & x : th) {
            x.join();
        }
    }

    std::vector<llama_token> out;
    out.reserve(text.size() / 4);
    size_t emitted = 0;
    for (int item : plan) {
        if (item < 0) {
            out.push_back(sep_id);
            continue;
        }
        const unsigned c = (n_threads <= 1) ? 0 : chunk_of((size_t) item);
        const size_t base = chunk_begin(c);
        const auto & r = ranges[c][(size_t) item - base];
        out.insert(out.end(), part[c].begin() + r.first, part[c].begin() + r.second);
        ++emitted;
    }
    if (progress) {
        progress->store(text.size(), std::memory_order_relaxed);
    }
    return out;
}

int main(int argc, char ** argv) {
    // keep partial output visible if the run fails
    setvbuf(stdout, nullptr, _IONBF, 0);
    setlocale(LC_NUMERIC, "C");
    demo_args args;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char * name) -> std::string {
            if (i + 1 >= argc) {
                fprintf(stderr, "[demo] missing value for %s\n", name);
                exit(1);
            }
            return argv[++i];
        };
        if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (a == "-v" || a == "--verbose") { args.verbose = true; }
        else if (a == "--model")       { args.model     = next("--model"); }
        else if (a == "--pretrain")    { args.pretrain  = true; }
        else if (a == "--config")      { args.config    = next("--config"); }
        else if (a == "--file")        { args.corpus    = next("--file"); }
        else if (a == "--device")      { args.device    = next("--device"); }
        else if (a == "--out")         { args.out       = next("--out"); }
        else if (a == "--save-dir")    { args.save_dir  = next("--save-dir"); }
        else if (a == "--no-save")     { args.no_save   = true; }
        else if (a == "--load-latest") { args.load_latest = true; }
        else if (a == "--infer")       { args.infer      = true; }
        else if (a == "--prompt")      { args.prompt     = next("--prompt"); }
        else if (a == "--n-predict")   { args.n_predict  = atoi(next("--n-predict").c_str()); }
        else if (a == "--temp")        { args.temp       = (float) atof(next("--temp").c_str()); }
        else if (a == "--top-p")       { args.top_p      = (float) atof(next("--top-p").c_str()); }
        else if (a == "--top-k")       { args.top_k      = atoi(next("--top-k").c_str()); }
        else if (a == "--seed")        { args.seed       = atoi(next("--seed").c_str()); }
        else if (a == "--repeat-penalty"){ args.rep_penalty = (float) atof(next("--repeat-penalty").c_str()); }
        else if (a == "--repeat-last-n") { args.rep_last_n  = atoi(next("--repeat-last-n").c_str()); }
        else if (a == "--epochs")      { args.epochs    = atoi(next("--epochs").c_str()); }
        else if (a == "--lr")          { args.lr        = (float) atof(next("--lr").c_str()); }
        else if (a == "--lr-min")      { args.lr_min       = (float) atof(next("--lr-min").c_str()); }
        else if (a == "--warmup")     { args.warmup       = atoi(next("--warmup").c_str()); }
    else if (a == "--decay")      { args.decay        = next("--decay"); }
    else if (a == "--decay-epochs"){ args.decay_epochs = atoi(next("--decay-epochs").c_str()); }
        else if (a == "--lr-scale")    { args.lr_scale     = (float) atof(next("--lr-scale").c_str()); }
        else if (a == "--wd")          { args.wd           = (float) atof(next("--wd").c_str()); }
        else if (a == "--ctx")         { args.n_ctx     = atoi(next("--ctx").c_str()); }
        else if (a == "--batch")       { args.n_batch   = atoi(next("--batch").c_str()); }
        else if (a == "--ubatch")      { args.n_ubatch  = atoi(next("--ubatch").c_str()); }
        else if (a == "--val-split")   { args.val_split = (float) atof(next("--val-split").c_str()); }
        else if (a == "--stride")      { args.stride    = atoi(next("--stride").c_str()); }
        else if (a == "--save-every")  { args.save_every = atoi(next("--save-every").c_str()); }
        else if (a == "--threads")     { args.n_threads = atoi(next("--threads").c_str()); }
        else {
            fprintf(stderr, "[demo] unknown option: %s\n", a.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    // guard the values the upstream parser/optimizer cannot cope with; without this a
    // tiny lr silently becomes 0 and ggml-opt aborts on GGML_ASSERT(alpha > 0)
    if (!(args.lr > 0.0f)) {
        fprintf(stderr, "[demo] --lr must be > 0 (got %g): ggml-opt asserts on a non-positive lr\n",
                (double) args.lr);
        return 1;
    }
    if (args.epochs <= 0) {
        fprintf(stderr, "[demo] --epochs must be >= 1 (got %d)\n", args.epochs);
        return 1;
    }

    // --config: build (if needed) and use the model described by the config file
    if (!args.config.empty() && !args.infer) {
        // assets/<config-stem>-random.gguf
        std::string stem = args.config;
        const size_t slash = stem.find_last_of("/\\");
        if (slash != std::string::npos) {
            stem = stem.substr(slash + 1);
        }
        const size_t dot = stem.find_last_of('.');
        if (dot != std::string::npos) {
            stem = stem.substr(0, dot);
        }
        const std::string out = "assets/" + stem + "-random.gguf";

        if (!file_exists(out)) {
            // the demo does not shell out to Python on purpose: the environment that runs the demo
            // may carry PYTHONHOME/PYTHONPATH that break a child interpreter, and the build step
            // deserves to be explicit and reproducible
            fprintf(stderr, "[demo] model for %s does not exist yet: %s\n",
                    args.config.c_str(), out.c_str());
            fprintf(stderr, "[demo] build it with:\n");
            fprintf(stderr, "[demo]   python tools/make-model.py --config %s\n", args.config.c_str());
            fprintf(stderr, "[demo] (trains the vocab first: python tools/train-tokenizer.py --help)\n");
            return 1;
        }

        printf("[demo] config : %s -> %s (random weights, vocab from the config)\n",
                args.config.c_str(), out.c_str());
        args.model    = out;
        args.pretrain = true;
    }

    // pretraining must not start from the released checkpoint: that would be a fine-tune.
    if (args.pretrain && !args.infer) {
        if (!args.config.empty()) {
            // the model came from --config (its own vocab, maybe not 32000): keep it
            printf("[demo] pretrain: random weights from %s -> first-step loss should be near ln(vocab_size)\n",
                    args.model.c_str());
        } else {
            if (!file_exists(args.pretrain_model)) {
                fprintf(stderr, "[demo] --pretrain needs a random-init GGUF: %s\n", args.pretrain_model.c_str());
                fprintf(stderr, "[demo] create it with: python fetch-assets.py --random-model --seed 1234\n");
                return 1;
            }
            if (args.model != args.pretrain_model) {
                printf("[demo] pretrain: using random init %s (was %s)\n",
                        args.pretrain_model.c_str(), args.model.c_str());
            }
            args.model = args.pretrain_model;
            printf("[demo] pretrain: random weights -> first-step loss should be near ln(32000) = 10.37\n");
        }
    } else if (!args.infer) {
        printf("[demo] fine-tune: starting from the weights in %s (use --pretrain for random init)\n",
                args.model.c_str());
    }

    // --load-latest: train on the newest model saved by a previous run. Resolved before
    // the argument vector is built, because that vector captures args.model by value.
    if (args.load_latest) {
        const std::string latest = newest_saved_model(args.save_dir);
        if (latest.empty()) {
            fprintf(stderr, "[demo] --load-latest: no .gguf in %s/\n", args.save_dir.c_str());
            llama_backend_free();
            return 1;
        }
        args.model = latest;
        printf("[demo] load-latest: %s\n", latest.c_str());
    }

    if (!file_exists(args.model) || (!args.infer && !file_exists(args.corpus))) {
        fprintf(stderr, "[demo] missing input:\n");
        if (!file_exists(args.model))  { fprintf(stderr, "[demo]   model  %s\n", args.model.c_str()); }
        if (!args.infer && !file_exists(args.corpus)) { fprintf(stderr, "[demo]   corpus %s\n", args.corpus.c_str()); }
        if (!file_exists(args.model)) {
            fprintf(stderr, "[demo] either fetch the released checkpoint: python fetch-assets.py\n");
            fprintf(stderr, "[demo] or build a model from a config (no checkpoint needed):\n");
            fprintf(stderr, "[demo]   python tools/make-model.py --config configs/minitrain-6l288-4k.json\n");
            fprintf(stderr, "[demo]   vulkan-minitrain --config configs/minitrain-6l288-4k.json --file <corpus>\n");
        }
        return 1;
    }

    common_init();
    llama_backend_init();

    // Resolve the requested device up front: the point of the demo is to be explicit
    // about where the training runs, instead of letting the scheduler pick.
    //
    // Note: upstream's --device only accepts *offload* devices and rejects the CPU
    // device outright (common/arg.cpp parse_device_list), where "none" means "no
    // offload". So a CPU run has to be passed through as --device none -ngl 0.
    if (args.device.empty()) {
        // no --device: pick the first non-CPU device, so a CUDA-only build works without flags
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(d) != GGML_BACKEND_DEVICE_TYPE_CPU) {
                args.device = ggml_backend_dev_name(d);
                break;
            }
        }
        if (args.device.empty()) {
            args.device = "none";   // CPU-only build
        }
    }

    ggml_backend_dev_t dev = ggml_backend_dev_by_name(args.device.c_str());
    bool cpu_only = false;
    if (dev == nullptr) {
        if (args.device == "none") {
            cpu_only = true;
        } else {
            fprintf(stderr, "[demo] device not found: %s\n", args.device.c_str());
            print_devices();
            llama_backend_free();
            return 1;
        }
    } else if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        cpu_only = true;
    }
    if (cpu_only) {
        printf("[demo] device : CPU only (no offload; --device %s)\n", args.device.c_str());
    } else {
        printf("[demo] device : %s (%s)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
    }

    // Let the upstream argument parser build the params instead of filling the struct
    // by hand: the parser applies defaults this demo would otherwise have to copy,
    // and some of those (e.g. a non-positive thread count) are not safe to skip.
    //
    // lr and val-split must be formatted with %g, not std::to_string: the latter keeps
    // only 6 decimals, so e.g. 3e-7 would be sent as "0.000000" and reach ggml-opt as 0.
    char lr_buf[32];
    char vs_buf[32];
    snprintf(lr_buf, sizeof(lr_buf), "%.10g", (double) args.lr);
    snprintf(vs_buf, sizeof(vs_buf), "%.10g", (double) args.val_split);

    // Training correctness: the K/V projections only receive gradients when n_ubatch == n_ctx
    // (llama warns "the K and V projections will not receive gradients" otherwise). Since
    // llama.cpp also clamps n_ubatch to n_batch, align both to the requested context size so
    // that e.g. --ctx 1024 works as intended without silently degrading the training.
    if (!args.infer && args.n_ubatch != args.n_ctx) {
        if (args.n_batch  < args.n_ctx) args.n_batch  = args.n_ctx;
        args.n_ubatch = args.n_ctx;
        printf("[demo] batch  : ubatch aligned to n_ctx (%d) so the K/V projections get gradients\n",
                args.n_ubatch);
    }

    std::vector<std::string> opt = {
        "vulkan-minitrain", "-m", args.model,
        "-c",  std::to_string(args.n_ctx),
        "-b",  std::to_string(args.n_batch),
        "-ub", std::to_string(args.n_ubatch),
        "-ngl", cpu_only ? "0" : "999",
        "--device", cpu_only ? "none" : args.device,
        "--epochs", std::to_string(args.epochs),
        "-lr", lr_buf,
        "--val-split", vs_buf,
    };
    // -f is only needed for training; upstream's parser opens the file, so inference
    // mode must not pass it (otherwise a missing corpus aborts the run)
    if (!args.infer) { opt.push_back("-f"); opt.push_back(args.corpus); }
    if (args.n_threads > 0) { opt.push_back("-t");  opt.push_back(std::to_string(args.n_threads)); }
    // save by default: --out wins, otherwise an auto-named file in --save-dir
    std::string out_path = args.out;
    if (!args.no_save && out_path.empty()) {
        out_path = join_path(args.save_dir, make_auto_name(args));
    }
    if (!out_path.empty()) { opt.push_back("-o"); opt.push_back(out_path); }
    if (args.verbose)       { opt.push_back("--verbose"); }

    std::vector<char *> optv;
    optv.reserve(opt.size());
    for (auto & s : opt) { optv.push_back(s.data()); }

    common_params params;
    params.escape = false;
    if (!common_params_parse((int) optv.size(), optv.data(), params, LLAMA_EXAMPLE_FINETUNE)) {
        fprintf(stderr, "[demo] argument parsing failed\n");
        llama_backend_free();
        return 1;
    }

    // learning-rate schedule / weight decay: these live on params.lr (struct lr_opt) and
    // upstream only exposes -wd as a CLI flag, so set the fields directly after parsing.
    if (args.lr_min > 0.0f)      { params.lr.lr_min = args.lr_min; }
    if (args.decay_epochs > 0)   { params.lr.decay_epochs = (float) args.decay_epochs; }
    if (args.lr_scale > 0.0f)    { params.lr.scale_epoch = args.lr_scale; }
    if (args.wd >= 0.0f)         { params.lr.wd = args.wd; }

    // training needs writable weights and f32 KV (OUT_PROD has no f16 backward path)
    if (params.load_mode != LLAMA_LOAD_MODE_NONE) { params.load_mode = LLAMA_LOAD_MODE_NONE; }
    params.cache_type_k = GGML_TYPE_F32;
    params.cache_type_v = GGML_TYPE_F32;

    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    if (!llama_init) {
        fprintf(stderr, "[demo] failed to initialise the model/context\n");
        llama_backend_free();
        return 1;
    }
    llama_model * model = llama_init->model();
    llama_context * ctx = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "[demo] failed to load the model: %s\n", params.model.path.c_str());
        llama_backend_free();
        return 1;
    }

    {
        char desc[256] = {0};
        llama_model_desc(model, desc, sizeof(desc));
        printf("[demo] model  : %s\n", desc);
        printf("[demo] size   : %.1f MiB, %lld params, n_layer %d, n_embd %d\n",
               llama_model_size(model) / 1048576.0,
               (long long) llama_model_n_params(model),
               llama_model_n_layer(model),
               llama_model_n_embd(model));
        printf("[demo] train  : n_ctx %d, batch %d, ubatch %d, lr %.2g, epochs %u, threads %d\n",
               params.n_ctx, params.n_batch, params.n_ubatch, (double) params.lr.lr0,
               params.lr.epochs, params.cpuparams.n_threads);
        if (params.lr.lr_min > 0.0f || params.lr.wd > 0.0f) {
            printf("[demo] sched  : lr_min %.2g, decay_epochs %.0f, lr_scale %.2g, wd %.2g\n",
                   (double) params.lr.lr_min, (double) params.lr.decay_epochs,
                   (double) params.lr.scale_epoch, (double) params.lr.wd);
        }
    }

    // inference mode: load the model, generate text on the chosen backend, no training
    if (args.infer) {
        const int rc = run_inference(ctx, model, args);
        llama_backend_free();
        return rc;
    }

    // Tokenization is single-threaded and takes ~0.4 s per MB of corpus (150 s for a 390 MB
    // slice), with nothing printed while it runs. A heartbeat makes clear that it is working,
    // not stuck -- and the per-MB rate is measured, so the estimate is honest.
    printf("[demo] phase  : tokenizing the corpus (single-threaded, roughly 0.4 s per MB) ...\n");
    fflush(stdout);
    const int64_t t_tok0 = ggml_time_ms();
    std::atomic<bool> tok_done{false};
    std::atomic<size_t> tok_pos{0};
    const size_t corpus_bytes = params.prompt.size();
    const size_t corpus_mb = corpus_bytes / (1024 * 1024) + 1;
    std::thread tok_watch([&tok_done, &tok_pos, corpus_bytes, corpus_mb, t_tok0]() {
        while (!tok_done.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::seconds(10));
            if (tok_done.load(std::memory_order_relaxed)) {
                break;
            }
            const double el = (ggml_time_ms() - t_tok0) / 1000.0;
            const size_t done_b = tok_pos.load(std::memory_order_relaxed);
            if (done_b > 0 && corpus_bytes > 0) {
                const double frac = (double) done_b / (double) corpus_bytes;
                const double eta  = frac > 0.01 ? el * (1.0 - frac) / frac : 0.0;
                printf("[demo]   ...   still tokenizing: %.0f s elapsed, %.0f%% done, ETA ~%.0f s\n",
                       el, 100.0 * frac, eta);
            } else {
                printf("[demo]   ...   still tokenizing: %.0f s elapsed of ~%.0f s (%zu MB corpus)\n",
                       el, 0.4 * (double) corpus_mb, corpus_mb);
            }
            fflush(stdout);
        }
    });
    std::vector<llama_token> tokens = tokenize_corpus(ctx, params.prompt, "<|endoftext|>", &tok_pos);
    tok_done.store(true, std::memory_order_relaxed);
    tok_watch.join();
    const int64_t t_tok1 = ggml_time_ms();
    if (tokens.size() < (size_t) (llama_n_ctx(ctx) + 1)) {
        fprintf(stderr, "[demo] corpus too short: %zu tokens for n_ctx %d\n",
                tokens.size(), llama_n_ctx(ctx));
        llama_backend_free();
        return 1;
    }

    // half-overlapping windows of n_ctx tokens; labels are the input shifted by one
    // sliding windows of n_ctx tokens; labels are the input shifted by one.
    // stride defaults to n_ctx/2 (half overlap): each token then appears in two windows, which
    // doubles the optimizer steps per epoch. --stride n_ctx halves that for the same coverage.
    int stride = args.stride > 0 ? args.stride : (int) llama_n_ctx(ctx) / 2;
    if (stride > (int) llama_n_ctx(ctx)) {
        printf("[demo] stride  : %d > n_ctx %d, clamping\n", stride, (int) llama_n_ctx(ctx));
        stride = (int) llama_n_ctx(ctx);
    }
    printf("[demo] phase  : building the sliding-window dataset (%zu tokens) ...\n", tokens.size());
    fflush(stdout);
    const int64_t t_ds0 = ggml_time_ms();
    ggml_opt_dataset_t dataset = common_opt_dataset_init(ctx, tokens, stride);
    const int64_t t_ds1 = ggml_time_ms();

    const int64_t ndata       = ggml_opt_dataset_ndata(dataset);
    const int64_t idata_split = (int64_t) (ndata * (1.0f - params.val_split));
    printf("[demo] tokens : %zu -> ndata %lld (train %lld / val %lld), stride %d\n",
           tokens.size(), (long long) ndata, (long long) idata_split,
           (long long) (ndata - idata_split), stride);
    printf("[demo] time   : tokenize %.1f s, dataset %.1f s\n",
           (t_tok1 - t_tok0) / 1000.0, (t_ds1 - t_ds0) / 1000.0);
    fflush(stdout);

    // step-level schedule: optimizer steps = train windows x windows per context x epochs
    const int64_t steps_per_dp = std::max<int64_t>(1, llama_n_ctx(ctx) / params.n_ubatch);
    const int64_t steps_epoch  = idata_split * steps_per_dp;

    lr_step_schedule sched;
    sched.total_steps = std::max<int64_t>(1, steps_epoch * (int64_t) params.lr.epochs);
    sched.lr0         = params.lr.lr0;
    sched.lr_min      = args.lr_min > 0.0f ? args.lr_min : 0.1f * params.lr.lr0;
    sched.wd          = args.wd > 0.0f ? args.wd : 0.0f;
    sched.decay       = args.decay == "off" ? 0 : (args.decay == "linear" ? 2 : 1);
    sched.warmup      = args.warmup < 0
        ? std::min<int64_t>(std::max<int64_t>(10, sched.total_steps / 100),
                            std::max<int64_t>(1, sched.total_steps / 10))
        : (int64_t) args.warmup;
    if (sched.warmup >= sched.total_steps) {
        sched.warmup = sched.total_steps / 10;
    }
    const bool lr_trace = std::getenv("MINITRAIN_LR_TRACE") != nullptr;

    if (sched.decay == 0) {
        printf("[demo] sched  : %lld steps, warmup %lld, constant lr %.3g, wd %.2g\n",
               (long long) sched.total_steps, (long long) sched.warmup,
               (double) sched.lr0, (double) sched.wd);
    } else {
        printf("[demo] sched  : %lld steps, warmup %lld, %s decay %.3g -> %.3g, wd %.2g\n",
               (long long) sched.total_steps, (long long) sched.warmup,
               sched.decay == 1 ? "cosine" : "linear",
               (double) sched.lr0, (double) sched.lr_min, (double) sched.wd);
    }

    struct llama_opt_params lopt_params{
        /*n_ctx_train     =*/ 0,
        /*param_filter    =*/ llama_opt_param_filter_all,
        /*param_filter_ud =*/ nullptr,
        /*get_opt_pars    =*/ lr_sched_pars,
        /*get_opt_pars_ud =*/ &sched,
        /*optimizer_type  =*/ params.optimizer,
    };
    printf("[demo] phase  : initialising the optimizer and the first graph "
           "(Vulkan pipelines compile here, so the GPU can idle a little longer) ...\n");
    fflush(stdout);
    const int64_t t_opt0 = ggml_time_ms();
    llama_opt_init(ctx, model, lopt_params);
    const int64_t t_opt1 = ggml_time_ms();
    printf("[demo] time   : opt init %.1f s  (startup total %.1f s)\n",
           (t_opt1 - t_opt0) / 1000.0,
           (t_tok1 - t_tok0 + t_ds1 - t_ds0 + t_opt1 - t_opt0) / 1000.0);
    printf("[demo] phase  : training starts now; the first steps also compile shaders on the GPU\n");
    fflush(stdout);

    ggml_opt_result_t result_train = ggml_opt_result_init();
    ggml_opt_result_t result_eval  = ggml_opt_result_init();

    // periodic checkpoints: safe for multi-hour unattended runs
    if (args.save_every > 0 && !out_path.empty()) {
        static checkpoint_state ckpt;
        ckpt.model     = model;
        ckpt.every_s   = args.save_every * 60;
        ckpt.base_path = out_path.substr(0, out_path.size() - std::string(".gguf").size());
        g_ckpt = &ckpt;
        printf("[demo] ckpt   : every %d min -> %s_step<N>.gguf\n", args.save_every, ckpt.base_path.c_str());
    }

    double loss_first = -1.0;
    double loss_last  = -1.0;

    printf("\n");
    for (unsigned epoch = 0; epoch < params.lr.epochs; ++epoch) {
        params.lr.epoch = epoch;

        const auto t0 = std::chrono::steady_clock::now();
        llama_opt_epoch(ctx, dataset, result_train, result_eval, idata_split,
                        epoch_callback_with_checkpoints,
                        ggml_opt_epoch_callback_progress_bar);
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        int64_t nd_train = 0;
        int64_t nd_eval  = 0;
        double  loss_train = 0.0, acc_train = 0.0;
        double  loss_eval  = 0.0, acc_eval  = 0.0;
        double  unc        = 0.0;

        ggml_opt_result_ndata   (result_train, &nd_train);
        ggml_opt_result_loss    (result_train, &loss_train, &unc);
        ggml_opt_result_accuracy(result_train, &acc_train, &unc);
        ggml_opt_result_ndata   (result_eval, &nd_eval);
        ggml_opt_result_loss    (result_eval, &loss_eval, &unc);
        ggml_opt_result_accuracy(result_eval, &acc_eval, &unc);

        if (loss_first < 0.0) {
            loss_first = loss_train;
        }
        loss_last = loss_train;

        printf("[demo] epoch %u: train loss %.5f acc %.2f%% (%lld tokens) | "
               "val loss %.5f acc %.2f%% (%lld tokens) | %.2f s | %.0f tok/s\n",
               epoch, loss_train, 100.0 * acc_train, (long long) nd_train,
               loss_eval, 100.0 * acc_eval, (long long) nd_eval,
               secs, nd_train / (secs > 0.0 ? secs : 1.0));

        ggml_opt_result_reset(result_train);
        ggml_opt_result_reset(result_eval);
    }

    if (params.lr.epochs > 1) {
        printf("\n[demo] summary: train loss %.5f -> %.5f over %u epoch(s)\n",
               loss_first, loss_last, params.lr.epochs);
    } else if (g_first_step_loss >= 0.0) {
        printf("\n[demo] summary: train loss %.5f (step 1) -> %.5f (end of the single epoch)\n",
               g_first_step_loss, loss_last);
    } else {
        printf("\n[demo] summary: train loss %.5f after %u epoch(s)\n",
               loss_last, params.lr.epochs);
    }

    if (!params.out_file.empty()) {
        const bool auto_named = args.out.empty();
        if (auto_named && !make_dir(args.save_dir)) {
            fprintf(stderr, "[demo] cannot create %s\n", args.save_dir.c_str());
        }
        printf("[demo] saving : %s\n", params.out_file.c_str());
        llama_model_save_to_file(model, params.out_file.c_str());
        report_save_dir(auto_named ? args.save_dir : std::string(), params.out_file);
    }

    printf("[demo] placement: set GGML_SCHED_DEBUG=2, rerun with --verbose and keep the\n");
    printf("[demo]            log, then count the per-node backends to see which ops\n");
    printf("[demo]            fell back to the CPU (see docs/APPLY-PATCH.md).\n");

    ggml_opt_result_free(result_train);
    ggml_opt_result_free(result_eval);
    ggml_opt_dataset_free(dataset);
    llama_backend_free();

    return 0;
}
