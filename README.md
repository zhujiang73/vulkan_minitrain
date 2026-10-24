# vulkan-minitrain

**用 Vulkan 从零训练一个小语言模型。**

## 1. 这是什么：Vulkan 训练 + 8.34 M 参数的小模型

[Vulkan](https://www.vulkan.org/) 是跨厂商的 GPU 计算/图形 API：NVIDIA、AMD、Intel 的显卡，
Windows / Linux / Android 都能跑。
**这个项目就是实验只用 Vulkan，把一个语言模型从随机初始化一路训到会说人话。**

能。下面是实测结果（单张 RTX 4060 Ti，从零训练约 511 M token 之后，val 1.79）：

![从零训练的 val 曲线](docs/figures/fig-fromscratch.png)

训练栈不是自己从零写的，而是 **vendor 进来的 llama.cpp**（裁剪版在 `src/llama_train/`）。
上游的 Vulkan 后端推理早已可用，但训练路径上有两处会拖慢甚至打断训练，本项目已把修复**合并进源码树**：

| # | 改动 | 作用 |
|---|---|---|
| 1 | `ggml.c`：给 `MUL_MAT` 反向的两处 `OUT_PROD` 补 `ggml_cont()` | Vulkan 要求 `OUT_PROD` 输入连续，否则**静默回落到 CPU**（实测 60.9% 的权重梯度） |
| 2 | `out_prod.comp`：重写为**共享内存分块 GEMM**（64×64 tile / 256 线程 / 每线程 4×4 寄存器块 / padding 防 bank 冲突） | `OUT_PROD` GPU 时间 29.17 → **11.63 ms**，端到端吞吐 **9,944 → 15,088 tok/s（1.52×）** |

训练数值与 CPU 参考实现一致（差异只在 f32 累加顺序），所以这不是"能跑但不可信"的演示。

### 实验用的模型架构

刻意选小：TinyStories 这类简单语料不需要大模型，小模型能在单卡上几小时训完，
于是"改一行着色器 → 跑一轮 → 看 loss"的循环才做得起来。

| 项 | 取值 |
|---|---|
| 架构 | LLaMA 式 decoder-only（RMSNorm + RoPE + SwiGLU） |
| 层数 / 宽度 / 头数 | **6 层 / 288 维 / 6 头**（MHA，非 GQA） |
| FFN 宽度 | 768 |
| 上下文 | 1024 |
| 词表 | **4,096**（从语料自训的字节级 BPE，见 §3） |
| 精度 | 全 **f32**（llama.cpp 没有 f16/量化的反向路径） |
| 参数量 / 体积 | **8.34 M 参数 / 31.9 MiB** |
| 配置 | [`configs/minitrain-6l288-4k.json`](configs/minitrain-6l288-4k.json) |

---

## 2. 准备数据集

```powershell
git clone <this repo> && cd vulkan-minitrain
python fetch-assets.py      # 下载 TinyStories 文本 + 可选 wikitext-2 跨域语料
```

下载前会**并行探测两个镜像**（`hf-mirror.com` 与 `huggingface.co`），谁先答应用谁（`--mirror` 可强制指定）；
下载带**断点续传**（HTTP Range，从 `assets/<文件名>.part` 接着下）、**单行进度条**（速度 + ETA）
和 **sha256 校验**（`TinyStories-train.txt` / `TinyStories-valid.txt` 都钉了固定值；`--no-verify` 可跳过）。
被 Ctrl-C 或断网中断后重跑即可续传。

`fetch-assets.py` 会准备好：

| 文件 | 说明 |
|---|---|
| `assets/TinyStories-train.txt` | TinyStories 全量训练文本（约 **1.92 GB**） |
| `assets/TinyStories-valid.txt` | 验证文本 |
| `assets/train-small.raw` | 小样例语料（demo 默认语料，用来快速冒烟） |
| `assets/wikitext-2-raw/` | 可选的跨域语料（用来验证"只会讲故事"的现象） |

> 1.92 GB 语料用本项目当前 4,096 词表分词后约 **480 M token**（旧词表口径是 522 M，两者的 token 大小不同）。
> 训练时语料会被切成 1024 token 的窗口，步长由 `--stride` 控制：默认 512（一半重叠），
> `--stride 1024` 则不重叠、步数减半（本项目正式跑用后者）。

---

## 3. 训练词表，然后把数据集分片

### 3.1 从语料训练 4K 词表

不借任何现成词表 —— 词表从**完整的全量语料**里学出来（字节级 BPE，纯标准库实现）。
这一步读的是未分片的大文件，让词表见到全部文本分布。

> **切分口径**：默认用整词切分（`--split word`），GGUF 里**不写** `tokenizer.ggml.pre`，让 llama.cpp
> 走它的 `default` 预切分。实测这是最优组合：
>
> | 100 MB / 1 轮 | 字节/token | val | bits/byte |
> |---|---|---|---|
> | **整词切分 + 不写 `pre`（默认）** | 3.777 | **4.44059** | **1.6966** |
> | gpt-2 切分 + `pre=gpt-2` | 3.995 | 4.99442 | 1.8034 |
>
> 390 MB 切片上的趋势一致（`gpt-2` 慢 10%）。`--split gpt2` / `--pre gpt-2` 仍然保留，用于实验，
> 但代价是加载时会打印 `missing pre-tokenizer` 警告的那条路径在实测中**反而更差**，所以默认不启用。
> 脚本会自动把 `<|endoftext|>` 保留为单个特殊 token（否则 gpt-2 切分会把它拆成三段，故事分隔符就丢了）。

```powershell
python tools\train-tokenizer.py --corpus assets\TinyStories-train.txt `
    --out-dir assets\vocab-4k --vocab-size 4096
```

1.92 GB 语料约 **十几分钟**，产出 `assets/vocab-4k/vocab.json` + `merges.txt`（4,096 token / 3,837 merges）。

### 3.2 按配置生成随机初始化的模型

```powershell
python tools\make-model.py --config configs\minitrain-6l288-4k.json
```

按 §1 的架构写出一个 **f32 GGUF**（随机权重：norm=1.0，其余 `N(0, 0.02)`），
并内嵌上面那份 4K 词表 → `assets/minitrain-6l288-4k-random.gguf`（31.9 MiB）。

确认它确实是随机初始化（可选，但建议跑一次）：

```powershell
python tools\check-random-init.py --model assets\minitrain-6l288-4k-random.gguf `
    --config configs\minitrain-6l288-4k.json
```

它检查三件事：所有 `*_norm.weight` 恰好为 1.0、其余张量 `N(0, 0.02)`、同种子重建逐字节相同。
**训练时的判据**：首个 loss 应接近 `ln(4096) = 8.32`（若看到 1~2，说明加载的是训好的模型）。

### 3.3 为什么必须分片

训练数据集在内存里是 **tokens + inputs + labels 三份**（`ggml_opt_dataset` 的布局），
所以一次加载 1.92 GB 文本约需 **19 GB** 内存，会被 OOM 杀掉。

注意顺序：**词表训练用完整的全量语料**（见 §3.1，不分片），分片只发生在"训练"这一步。

切成 **390 MB 左右的 5 片**后，每片峰值内存约 3.4 GB，串行训练即可跑完全量语料：

```powershell
python tools\make-slices.py --slice-mb 390
#  assets/TinyStories-train.txt  →  assets/ts-full-1.raw … ts-full-5.raw（390/390/390/390/275 MB）
```

脚本是纯 Python、跨平台，已存在的分片会自动跳过（`--force` 强制重写）；`--corpus` / `--out-dir` / `--prefix`
可以改输入输出位置。

---

## 4. 从零训练：5 片串接，约 5 小时出一个会说人话的模型

第 1 片用随机初始化的模型，其后每片**从上一片的产物继续**（`--load-latest`）；
学习率恒定（`--decay off`），每 30 分钟存一个 checkpoint 以防中断。

五条命令，**按顺序各跑一条**（每条都是完整命令，换 shell 也不变；第 1 条用 `--config`，后四条用 `--load-latest`）：

```powershell
.\build\bin\Release\vulkan-minitrain.exe --config configs\minitrain-6l288-4k.json --file assets\ts-full-1.raw --device Vulkan0 --epochs 1 --lr 1e-6 --stride 1024 --decay off --save-every 30
.\build\bin\Release\vulkan-minitrain.exe --load-latest --file assets\ts-full-2.raw --device Vulkan0 --epochs 1 --lr 1e-6 --stride 1024 --decay off --save-every 30
.\build\bin\Release\vulkan-minitrain.exe --load-latest --file assets\ts-full-3.raw --device Vulkan0 --epochs 1 --lr 1e-6 --stride 1024 --decay off --save-every 30
.\build\bin\Release\vulkan-minitrain.exe --load-latest --file assets\ts-full-4.raw --device Vulkan0 --epochs 1 --lr 1e-6 --stride 1024 --decay off --save-every 30
.\build\bin\Release\vulkan-minitrain.exe --load-latest --file assets\ts-full-5.raw --device Vulkan0 --epochs 1 --lr 1e-6 --stride 1024 --decay off --save-every 30
```

> **启动阶段**：程序会把整段语料先分词再建数据集，这一步是**单线程**的，约 **0.4 s/MB**
> （390 MB 分片 ≈ 150 秒，期间 GPU 空闲；另有约 10 秒一次性初始化）。程序会每 10 秒打印一次 `still tokenizing ...`，训练真正开始时
> 会打印 `training starts now`；之后的耗时才是 GPU 训练时间。

实测（RTX 4060 Ti / Vulkan0）：

| 片 | token | train loss | **验证集 loss** | acc(train/val) | 耗时 |
|---|---|---|---|---|---|
| 1 | 108.6 M | 4.43886 | 3.53136 | 23.4% / 31.7% | 65 分钟 |
| 2 | 108.7 M | 2.96357 | 2.60664 | 39.1% / 44.2% | 65 分钟 |
| 3 | 108.6 M | 2.37786 | 2.17682 | 47.7% / 50.9% | 65 分钟 |
| 4 | 108.6 M | 2.05165 | 1.97437 | 52.7% / 53.9% | 64 分钟 |
| 5 | 76.6 M | 1.88005 | **1.78994** | 55.4% / 56.9% | 45 分钟 |
| **合计** | **511.1 M** | — | — | — | **5.08 小时（26,545 tok/s）** |

最终模型会说的话（temp 0.8 / seed 7）：

> Once upon a time, there was a little girl named Lily. She loved playing in the park with her friends. One day,
> Lily's mom brought her a new video in the park. It was a special movie that made music. Lily loved to play
> dress-up in her backyard.

> Tom and his dog went to the park. He saw that the dog was not nice. He looked at the slide and made a lot of
> noise. It was big and noisy. "Hey, ha, what are you doing?" Tom asked. He had an idea.

自己生成一段：

```powershell
.\build\bin\Release\vulkan-minitrain.exe --load-latest --infer --prompt "long  long  ago" --seed  2048  --n-predict 150 --temp 0.8
```

**两点必须知道**：

1. 片边界会**重置优化器动量**（AdamW 的 m/v），所以逐片 val 不是"单条调度下的收敛曲线"，
   但作为"从零到什么程度"的判据是有效的；
2. 每片的验证集是该片自己的 5% 留出，片与片之间严格"先训后测"。

---

**构建与文档**：

```powershell
cmake -B build -G "Visual Studio 18 2026" -A x64
cmake --build build --config Release --target vulkan-minitrain 
```

    本项目的主要工作是在 deepseek harness 中完成的。
    
                          zhujiangmail@hotmail.com
                          
                                                    2026.10.24


许可证：[MIT](LICENSE)（本项目代码）；`src/llama_train/` 是 llama.cpp，同为 MIT，其许可证原文保留在
[`src/llama_train/LICENSE`](src/llama_train/LICENSE)。
