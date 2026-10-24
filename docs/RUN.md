# 编译完成后：手动运行训练程序

本文是"编译已经完成"之后的操作说明 —— 怎么手动把训练跑起来、参数怎么给、输出在哪、报错怎么办。

> **路径约定**：本文命令里的相对路径以**工作区根目录**为基准（即包含 `vulkan-minitrain\` 的那一层；llama.cpp 已合并进 `vulkan-minitrain\src\llama_train\`）。
> 演示程序自己产出的东西（可执行文件、模型、日志）都在 `vulkan-minitrain\` 下。

---

## 0. 最短路径

```powershell
cd vulkan-minitrain
.\build\bin\Release\vulkan-minitrain.exe --epochs 2
```

可执行文件固定在 **`vulkan-minitrain\build\bin\Release\vulkan-minitrain.exe`**
（Debug 构建则在 `build\bin\Debug\`）。

它和 llama/ggml 的 DLL（`llama.dll`、`llama-common.dll`、`ggml*.dll`）**放在同一个目录**，所以不需要改 PATH、也不需要拷贝任何东西 —— 在原地双击或命令行运行都可以。

## 1. 唯一要注意的坑：默认参数是**相对路径**

默认 `--model assets/stories15M.gguf`、`--file assets/train-small.raw` 是相对**当前工作目录**的。
**注意**：`assets/stories15M.gguf` 不再随仓库提供（`fetch-assets.py` 只取语料）——从零训练请用
`--config configs/minitrain-6l288-4k.json`，微调则需要你先自备一个 GGUF：

| 做法 | 结果 |
|---|---|
| `cd vulkan-minitrain` 再运行 | ✅ 默认参数即可 |
| 在别处运行、不给路径 | ❌ `[demo] missing input:` + `run fetch-assets.py first` |
| 在别处运行、给绝对路径 | ✅ 正常 |

不想切目录就显式给路径：

```powershell
& "$PWD\vulkan-minitrain\build\bin\Release\vulkan-minitrain.exe" `
    --model "$PWD\vulkan-minitrain\assets\stories15M.gguf" `
    --file  "$PWD\vulkan-minitrain\assets\train-small.raw" `
    --device Vulkan0 --epochs 2
```

## 2. 命令行参数

```
--model <path>    f32 GGUF 模型      (默认 assets/stories15M.gguf)
--file <path>     训练用纯文本        (默认 assets/train-small.raw)
--device <name>   后端设备            (默认 Vulkan0)
--epochs <n>      epoch 数            (默认 2)
--lr <f>          AdamW 学习率        (默认 1e-5)
--ctx <n>         上下文长度          (默认 1024，等价上游 -c；训练时 batch/ubatch 自动对齐到它)
--batch <n>       逻辑批量            (默认 512，等价 -b)
--ubatch <n>      物理批量            (默认 512，等价 -ub)
--val-split <f>   验证集比例          (默认 0.05)
--threads <n>     CPU 线程            (默认 0 = 交给上游按物理核数决定)
--out <path>      保存微调后的模型
-v, --verbose     放开日志阈值（**落点分析必需**）
-h, --help        帮助
```

```powershell
cd vulkan-minitrain
$exe = ".\build\bin\Release\vulkan-minitrain.exe"

# 多跑几轮、调大学习率、存模型
& $exe --epochs 5 --lr 5e-5 --out tuned.gguf

# 纯 CPU 跑（本机构建里有两个设备：Vulkan0 与 CPU）
& $exe --device CPU --epochs 2

# 列出可用设备：随便给个不存在的设备名，它会报错并列出可用设备
& $exe --device xxx
```

> **三个限制**
> ① 默认构建只有 **`Vulkan0` 和 `CPU`**（我特意把 `GGML_CUDA=OFF`，演示只要 Vulkan、构建最快）。想用 `--device CUDA0` 得先用 `-DGGML_CUDA=ON` 重新配置并编译（CUDA 部分很慢，约 40 分钟）；要对比 CUDA 也可以直接用主实验留下的 `lb\bin\llama-finetune.exe`。
> ② "列设备"发生在**输入文件检查之后**，所以要么在项目目录里跑上面那条，要么补上 `--model/--file` 绝对路径。
> ③ **纯 CPU 用 `--device CPU` 或 `--device none`**，两者等价（实测：`--device CPU` 19.13 s、`--device none` 20.01 s，loss 都是 8.13723）。
> 注意上游的 `--device` **只接受非 CPU 的 offload 设备**：如果直接把 `CPU` 传给上游解析器，会报
> `error while handling argument "--device": invalid device: CPU`（`common/arg.cpp` 的 `parse_device_list` 显式拒绝 `GGML_BACKEND_DEVICE_TYPE_CPU`）。
> 演示内部会把它映射成上游认识的 `--device none -ngl 0`，所以直接写 `--device CPU` 是安全的。

## 3. 输出去哪儿了

| 内容 | 去向 |
|---|---|
| `[demo]` 开头的行（设备 / 模型 / 数据 / 每轮 loss 与吞吐 / 汇总） | **stdout** |
| 训练与验证的进度条（上游 `ggml_opt_epoch_callback_progress_bar`） | **stderr** |
| 微调后的模型 | 只有给了 `--out` 才写，一个约 94 MB 的 GGUF |

```powershell
cd vulkan-minitrain
$exe = ".\build\bin\Release\vulkan-minitrain.exe"

& $exe --epochs 2 > run.log            # 只留 [demo] 汇总
& $exe --epochs 2 *> run-full.log      # 连进度条一起留
```

**退出码**：`0` 正常；`1` 表示输入缺失 / 设备不存在 / 参数解析失败。

一次 1 epoch（Vulkan0）的实测输出：

> `train loss` / `val loss` / `acc` 以及进度条里 `loss=X±Y` 的 `±Y` 各是什么意思、为什么**只能用 val 判断好坏**，见 [TRAINING-SCALE.md](TRAINING-SCALE.md) 第 2 节「指标口径」。


```
[demo] device : Vulkan0 (NVIDIA GeForce RTX 4060 Ti)
[demo] model  : llama ?B (guessed) all F32
[demo] size   : 93.1 MiB, 24407712 params, n_layer 6, n_embd 288
[demo] train  : n_ctx 512, batch 512, ubatch 512, lr 1e-05, epochs 1, threads 6
[demo] tokens : 9890 -> ndata 36 (train 34 / val 2)
[demo] epoch 0: train loss 8.13686 acc 0.09% (17408 tokens) | val loss 8.28643 acc 0.09% (1024 tokens) | 8.65 s | 2012 tok/s
[demo] summary: train loss 8.13686 -> 8.13686 over 1 epoch(s)
```

> 只跑 1 个 epoch 的耗时会包含首轮约 7 秒的**一次性 Vulkan 管线创建**；想看稳态速度就跑 2 个 epoch 看第 2 个（约 1.8 s）。详见 [PLACEMENT-AND-PERF.md](PLACEMENT-AND-PERF.md) 第 3 节。

## 4. 想看"活干在哪"（落点）

```powershell
cd vulkan-minitrain
$env:GGML_SCHED_DEBUG = '2'
.\build\bin\Release\vulkan-minitrain.exe --epochs 1 --verbose > sched.log 2>&1
Remove-Item Env:\GGML_SCHED_DEBUG

python tools\placement-recount.py sched.log
```

**`--verbose` 不能省**：llama.cpp 的日志回调默认按 INFO 级过滤，会把 ggml 的 DEBUG 输出整段丢掉（只设 `GGML_SCHED_DEBUG` 时日志里 `## SPLIT` 与 `node #` 都是 0 行）。

期望结果：

| 源码树状态 | 落点统计输出 |
|---|---|
| 未打补丁 | `OUT_PROD: CPU 67 / GPU 43 -> 60.9% 的权重梯度在 CPU 上算` |
| 已打补丁 | `OUT_PROD: CPU 0 / GPU 3740 -> 0.0%` |

目前的源码树**已内置**该改动（`src/llama_train/`），上表的"已打补丁"就是默认状态；改动清单见 [APPLY-PATCH.md](APPLY-PATCH.md) 与 `src/llama_train/LOCAL-CHANGES.md`。

## 5. 常见报错对照

| 现象 | 原因 / 处理 |
|---|---|
| `[demo] missing input:` + `run fetch-assets.py first` | 不在项目目录且没给绝对路径；或资源没取（`python vulkan-minitrain\fetch-assets.py`） |
| `[demo] device not found: xxx` + 设备列表 | 设备名写错；本机可用的是 `Vulkan0` / `CPU`（列表里 CPU 那行会标 `[cpu, no offload]`） |
| `error while handling argument "--device": invalid device: CPU` | 上游 `--device` 不接受 CPU 设备（只收 offload 设备）。**当前的演示已把 `--device CPU` 映射成 `--device none -ngl 0`**；如果你在自己代码里直接调 `llama-finetune`/上游解析器，请改用 `--device none -ngl 0` |
| `[demo] corpus too short` | 语料 token 数不足一个 `n_ctx`：换更大的文本，或调小 `--ctx` |
| `[demo] failed to initialise the model/context` | 模型不是 f32、或文件损坏 —— 训练要求 **f32** GGUF |
| `[demo] failed to load the model: ...` | 同上；也检查模型路径 |
| 提示找不到 `ggml-vulkan.dll` 之类 | exe 被移走了 —— DLL 必须和 exe 同目录（`build\bin\Release\`） |
| 训练 loss 一直不降 | 学习率太小（默认 1e-5，1~2 个 epoch 变化本来就小）；试 `--lr 5e-5` 或 `--epochs 5` |
| 显存/内存不足 | 调小 `--ctx`、`--batch`、`--ubatch` |

## 6. 从零到跑起来（跨平台命令序列）

原来项目根目录有个 `run-demo.ps1` 把这几步串成一条命令，但它是 **Windows/PowerShell 专用**的，为了将来能移植到 Linux 已经**删除**。现在把流程显式写出来 —— 三步，Windows 与 Linux 只差构建命令的写法。

```powershell
# ① 取资源（跨平台，纯 Python 标准库；本机已有副本会直接复制，否则走 hf-mirror）
python vulkan-minitrain\fetch-assets.py

# ② 配置 + 构建（Windows）
cd vulkan-minitrain
$env:TrackFileAccess = 'false'
cmake -S . -B build -G "Visual Studio 18 2026" -A x64
cmake --build build --config Release --target vulkan-minitrain -- `
      /p:TrackFileAccess=false /p:BuildInParallel=false

# ③ 运行
.\build\bin\Release\vulkan-minitrain.exe --epochs 2
```

```bash
# Linux（同一份 CMakeLists，无需改动）
cd vulkan-minitrain
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      # 默认使用 src/llama_train，无需传路径
cmake --build build            # 单配置生成器：不要加 --config
./build/bin/Release/vulkan-minitrain --epochs 2
```

- 单配置生成器（Ninja / Makefiles）下，可执行文件同样落在 `build/bin/<CMAKE_BUILD_TYPE>/`，所以要显式给 `-DCMAKE_BUILD_TYPE=Release`。
- Linux 上 `GGML_VULKAN=ON` 需要 Vulkan SDK（`glslc` 用于编译那 1566 个着色器变体）；CUDA 可选 `-DGGML_CUDA=ON`。
- `TrackFileAccess` / `BuildInParallel` 这些是 **MSBuild 专有开关**，Linux 上直接不加。


## 6.x 训练产物与续训（`output/`）

模型默认保存到项目内的 `output/`，文件名形如
`stories15M_tinystories-10mb_ep1_lr1e-06_20261020-174125.gguf`（基座_语料_轮数_学习率_时间戳）。

```powershell
# 默认保存（不给 --out 就自动命名进 output/）
.\build\bin\Release\vulkan-minitrain.exe --file .\assets\tinystories-10mb.raw --epochs 1 --lr 1e-6

# 输出示例
#   [demo] saving : output/stories15M_tinystories-10mb_ep1_lr1e-06_20261020-174125.gguf
#   [demo] output : 1 model(s) in output/, 93.8 MB total

# 续训：加载 output/ 里最新的模型
.\build\bin\Release\vulkan-minitrain.exe --load-latest --file .\assets\tinystories-10mb.raw --epochs 1 --lr 1e-6

# 不保存 / 存到指定位置
.\build\bin\Release\vulkan-minitrain.exe --no-save ...
.\build\bin\Release\vulkan-minitrain.exe --out D:\models\mine.gguf ...
```

- 每个 f32 模型约 **94 MB**；`output/*` 已被 `.gitignore` 忽略（目录本身保留）
- 目录里超过 **4 个**模型时会有 `[demo] WARN` 提醒清理，例如：
  `del output\*.gguf`，或只留最新：`Get-ChildItem output\*.gguf | Sort-Object LastWriteTime | Select-Object -SkipLast 1 | Remove-Item`
- 保存的是完整 f32 GGUF，可直接用 `--model output\xxx.gguf` 加载（数值会体现在下次训练的初始 loss 上）


## 6.y 推理模式（`--infer`）

模型训练/保存之后，用推理模式验证效果（不训练、不写盘）：

```powershell
# 基座模型
.\build\bin\Release\vulkan-minitrain.exe --infer --prompt "Once upon a time, there was a little" --n-predict 64 --temp 0
# 训练产物（output/ 里最新的）
.\build\bin\Release\vulkan-minitrain.exe --infer --load-latest --n-predict 64
# 指定文件
.\build\bin\Release\vulkan-minitrain.exe --infer --model output\xxx.gguf --n-predict 128 --temp 0.8 --top-p 0.95 --top-k 40 --seed 42
```

- 推理**不需要** `--file`（只有训练模式需要语料）
- `--temp 0` 为贪心采样；`--seed` 控制随机性，便于复现
- 输出末尾打印 `generated N token(s) in X s (Y tok/s)`
- 后端仍由 `--device` 决定；`--device none` 可跑纯 CPU 做对照

## 7. 用完整 wiki.train.raw 训练（10.4 MB，1 轮 ≈ 9 分钟）

默认的 `assets/train-small.raw` 只有 40 KB（34 步/轮），适合验证能不能跑。真正跑一次要用 **WikiText-2 完整训练集**：

### 7.1 准备语料

完整语料已经在工作区里（由主实验解压），不需要重新下载：

```powershell
# 应为 ~10.4 MB，2,827,025 token
Get-Item assets\wikitext-2-raw\wiki.train.raw

# 如果没有，用主实验留下的 zip 解压（zip 本身由 fetch-assets.py 下载）
Expand-Archive assets\wikitext-2-raw-v1.zip -DestinationPath assets -Force
```

**不需要手工切分**：演示程序按 `--ctx`（默认 512）自动开窗、按 `--val-split`（默认 0.05）切训练/验证。

### 7.2 先估算时间，再决定跑几轮

| 项 | 值 |
|---|---|
| token 数 | 2,827,025 |
| 每轮步数 | **10,488**（1 轮 = 把语料完整过一遍） |
| 每轮墙钟（Vulkan0） | **≈ 8.9 分钟**（首轮多 ~7 s 一次性管线创建） |
| 吞吐 | ~10,000 token/s = **每 token 0.1 ms**，与语料大小无关 |
| 纯 CPU 对照 | ≈ 11.5× 慢 → **每轮约 1.7 小时** |

时间预算就是用轮数乘：1 轮 9 分钟 · 2 轮 18 分钟 · 6 轮 54 分钟 · 9 轮 81 分钟。

### 7.3 命令

```powershell
cd vulkan-minitrain
$exe  = ".\build\bin\Release\vulkan-minitrain.exe"
$full = "assets\wikitext-2-raw\wiki.train.raw"   # python fetch-assets.py 会准备好

# ① 先跑 1 轮确认能跑通（9 分钟），看 val 是否正常
& $exe --file $full --epochs 1 --lr 1e-6

# ② 确认后再加轮：6 轮 = 54 分钟（实测性价比最优，val 5.312）
& $exe --file $full --epochs 6 --lr 1e-6

# ③ 想找平台期：9 轮 = 81 分钟（val 5.273，第 7 轮起已趋平）
& $exe --file $full --epochs 9 --lr 1e-6

# ④ 保存微调后的模型（约 94 MB GGUF）
& $exe --file $full --epochs 1 --lr 1e-6 --out .\stories15m-wikitext.gguf

# ⑤ 纯 CPU 对照（每轮约 1.7 小时，先想清楚）
& $exe --file $full --epochs 1 --lr 1e-6 --device CPU
```

> ⚠️ **学习率必须用小值**。完整语料每轮 1 万步，`1e-6` 才在稳定区；**上游默认的 `1e-5` 在这里从第 1 轮起就劣化**（val 5.73 → 7.82），`1e-4` 会直接发散。扫描数据见 [TRAINING-SCALE.md](TRAINING-SCALE.md) 第 5–6 节。

### 7.4 期望输出（1 轮实测）

```
[demo] device : Vulkan0 (NVIDIA GeForce RTX 4060 Ti)
[demo] model  : llama ?B (guessed) all F32
[demo] size   : 93.1 MiB, 24407712 params, n_layer 6, n_embd 288
[demo] train  : n_ctx 512, batch 512, ubatch 512, lr 1e-06, epochs 1, threads 6
[demo] tokens : 2827025 -> ndata 11041 (train 10488 / val 553)
[demo] epoch 0: train loss 6.00721 acc 0.16% (5369856 tokens) | val loss 5.72511 acc 0.19% (283136 tokens) | 540.50 s | 9935 tok/s
[demo] summary: train loss 6.00721 -> 6.00721 over 1 epoch(s)
```

各字段（`train` / `val` / `acc` / 进度条里的 `±`）含义见 [TRAINING-SCALE.md](TRAINING-SCALE.md) 第 2 节「指标口径」。

### 7.5 跑十几分钟以上的任务：放后台 + 落日志

```powershell
$exe  = ".\build\bin\Release\vulkan-minitrain.exe"
$full = "assets\wikitext-2-raw\wiki.train.raw"   # python fetch-assets.py 会准备好
$log  = "docs\fullcorpus-vulkan-6ep-lr1e-6.log"

& $exe --file $full --epochs 6 --lr 1e-6 2>&1 |
  Tee-Object -FilePath $log |
  Select-String -Pattern '^\[demo\] (device|train|tokens|epoch|summary)'
```

- `[demo]` 行在 **stdout**、进度条在 **stderr**，`2>&1` 一起收集
- 进度条每 10 步打一行，**1 轮约 1 万行**（6 轮约 17 MB 日志）
- **日志是 UTF-16**（PowerShell `Tee-Object` 的默认编码），用 Python 解析记得 `encoding="utf-16"`
- 想提取轮内轨迹：正则 `train: .*?data=(\d+)/(\d+) loss=([0-9.]+)`

### 7.6 注意事项

1. **一次只跑一个**：两个训练同时跑会互相抢 GPU，吞吐和 loss 读数都会失真。
2. **跑之前/期间别做构建**：MSBuild 会抢 CPU，污染 Vulkan 的 CPU 侧工作（连纯 CPU 那一列也会被带偏）。
3. **`--ctx` 直接影响步数与耗时**：默认 512 → 10,488 步/轮；改成 `--ctx 256` 步数翻倍（耗时约翻倍），`--ctx 128` 再翻倍。**但 128 才是模型声明的训练上下文**（见 [MODEL.md](MODEL.md) 第 6 节 ②）。
4. **显存不是瓶颈**：93 MB 模型 + 512 token 窗口，16 GB 卡远远够；瓶颈是逐算子启动开销（这也是 Vulkan 比 CUDA 慢的主因）。
5. **想看"活干在哪"**：加 `--verbose` 并设 `GGML_SCHED_DEBUG=2`，见 [APPLY-PATCH.md](APPLY-PATCH.md) 第 5 节。

### 7.7 同域语料：TinyStories（想看"真的在学"就跑这个）

`stories15M` 本身就是在 **TinyStories** 上训练的，用 WikiText 微调属于**跨域**，loss 会停在 5–6 附近。想看明显的学习效果，先取一份**去重后的同域切片**：

```powershell
cd vulkan-minitrain
$exe = ".\build\bin\Release\vulkan-minitrain.exe"

# 下载 18.55 MB 的 TinyStories valid 集 → 按 <|endoftext|> 切分 → 去重 → 落盘 10 MB
python fetch-assets.py --corpus tinystories --slice-mb 10
#  → 12,356 篇故事 / 10.00 MB / ~2.5M token（valid 集扫描结果：0 条重复）

# 同域微调（1 轮约 9 分钟，与完整 WikiText 语料同量级）
& $exe --file .\assets\tinystories-10mb.raw --epochs 1 --lr 1e-6
```

实测对照（都是 1 轮、`lr 1e-6`、约 11,000 步）：

| 语料 | train | **val** | acc | ppl = exp(val) |
|---|---|---|---|---|
| WikiText-2（跨域） | 6.007 | **5.725** | 0.16% | ~307 |
| **TinyStories（同域）** | **1.968** | **1.948** | **0.54%** | **~7** |

**val 差 3.78 nat、困惑度 307 → 7**，而吞吐完全一样（9,944 vs 9,935 tok/s）—— 差别只来自语料与模型的匹配度。

- 想快速看到"梯度下降在起作用" → 用 TinyStories 切片
- 想做学习率 / 轮数标定 → 用完整 WikiText 语料（量级与公开基准可比，见 [TRAINING-SCALE.md](TRAINING-SCALE.md)）
- 切片脚本会**精确去重**并丢弃过短残片：TinyStories 的 train/parquet 分片已知含约 **15% 完全重复**

#### 更大切片：备份整份 train.txt，或按 Range 取前缀

valid 集只有 22k 篇（18.55 MB），切片上限约 18 MB。需要长训练时用 **train 分片**（约 210 万篇，1.79 GiB）：

```powershell
# ① 下载整份 train.txt（1.79 GiB）→ assets/TinyStories-train.txt，带 size + sha256 校验
python fetch-assets.py --full-train

# ② 从本地这份大文件切 40 MB（秒级完成，不再联网）
python fetch-assets.py --corpus tinystories --source train --slice-mb 40
#  → assets/tinystories-train-40mb.raw（约 1,000 万 token，1 轮约 17 分钟）

# ③ 需要整轮训练时，把它切成 5 片（跨平台、跳过已存在的分片）
python tools\make-slices.py --slice-mb 390
```

- 本地没有大文件时，`--source train` 会退化成**按 HTTP Range 只下载需要的字节**（约 1.5 倍切片大小），再在**最后一个 `<|endoftext|>` 处截断**，所以不会出现半截故事。
- 校验值来自 HF 的 LFS oid：`sha256 = c5cf5e22…d401f`，期望大小 `1,924,281,556` 字节；脚本默认校验，`--no-verify` 可跳过。
- 这份 1.79 GiB 文件**只作本地数据源**，不要提交到 Git（见 `.gitignore` 建议）。

### 7.7.1 加载模型时的两条警告（已修复）

早期用 `tools/make-model.py` 生成的 GGUF 加载时会打印：

```
W load: missing pre-tokenizer type, using: 'default'
W load: GENERATION QUALITY WILL BE DEGRADED!
W load: control-looking token: 388 '<|endoftext|>' was not control-type; ...
```

原因与修法（`tools/make-model.py`）：

| 警告 | 缺失的元数据 | 现在写入 |
|---|---|---|
| missing pre-tokenizer type | `tokenizer.ggml.pre`（分词前的切分规则） | `"gpt-2"` —— 与自训 BPE 使用的 GPT-2 规则一致 |
| control-looking token ... | `<\|endoftext\|>` 的 token 类型 | 标为 **CONTROL(3)**，不再触发"疑似模型 bug"告警 |

配套改动：CONTROL 类型的 `<|endoftext|>` 在 `parse_special = false` 下会被拆成 **7 个普通 token**，
分隔符就学不到了。但**不能**简单地对整段语料用 `parse_special = true`：实测 390 MB 语料的分词会从
**150 s 退化到 10 分钟以上**（llama.cpp 在该模式下反复扫描特殊 token），表现为"永远卡在 still tokenizing"。

现在的做法是按分隔符切块（`tokenize_corpus()`）：每段故事用普通快速路径分词，再手工插入分隔符的 token id。
因为分隔符在参考编码里是**原子的单个 token**（没有 BPE 合并能跨过它），所以 token 序列与
`parse_special = true` **完全一致**——实测同一个 10 MB 切片都是 2,713,982 个 token——而 390 MB 只要 **135 s**。

顺带：分词心跳现在显示真实百分比与 ETA（按已处理字节数），不再用按 MB 估的粗略数字，
形如 `still tokenizing: 40 s elapsed, 31% done, ETA ~89 s`。

**分词是单线程的**（390 MB ≈ 130 s）。多线程版本实现过但**不安全**（≥4 线程崩溃、2 线程虽然快 1.86×
但 token 数与单线程差 200），因此默认关闭，只有设 `MINITRAIN_TOK_THREADS` 才会启用（实验性）。

**结论：保留"不写 `pre`"的默认路径（2026-10-24 实测）**

一开始判定根因是词表训练切分（整词）与 `pre=gpt-2` 不一致，于是重训了 4K 词表并对齐到 gpt-2。
但**放大到真实规模后结论反转**：

| 规模 | 整词 + 不写 `pre`（历史默认） | gpt-2 切分 + `pre=gpt-2` | 差 |
|---|---|---|---|
| 10 MB / 1 轮 | val 7.13672（2.7057 bits/byte） | val 7.13620（2.5565） | gpt-2 看似更好（早期噪声） |
| **100 MB / 1 轮** | val **4.44059（1.6966 bits/byte）** | val 4.99442（1.8034） | **默认好 5.9%** |
| **390 MB / 1 轮** | val **3.24165（1.2729）** | val 3.86312（1.3984） | **默认好 9.9%** |

390 MB 上还验证了词表本身的影响：新旧词表都配 `pre=gpt-2` 时 bits/byte 分别是 1.3984 / 1.4024（**基本相同**），
说明差异**几乎全部来自 `pre`**，与词表无关。

因此默认配置回退为：`--split word` + 不写 `tokenizer.ggml.pre`（`make-model.py --pre none`，默认值），
即"加载时会打印 missing pre-tokenizer 警告"的那条路径。警告是元数据提示，实测质量反而更好。
`--split gpt2` / `--pre gpt-2` 保留为实验选项。

> 历史注记：下面这段是当时（只做了 10 MB 测试时）的分析，保留以便理解当时为什么走了弯路。
把 `tools/train-tokenizer.py` 的切分改成 llama.cpp 的 gpt-2 正则（ASCII 转写）并**重训 4K 词表**后，
两边一致，实测（10 MB / 1 轮）：

| 组合 | tokens | 字节/token | val | acc | bits/byte |
|---|---|---|---|---|---|
| 旧词表（整词切分）+ `pre=gpt-2` | 2,712,576 | 3.866 | 7.15579 | 5.75% | 2.6706 |
| 旧词表 + 不写 `pre`（历史基线） | 2,755,584 | 3.805 | 7.13672 | 5.64% | 2.7057 |
| 新词表（gpt-2 切分，缺 `<\|endoftext\|>`） | 2,628,496 | 3.989 | 7.15816 | 5.89% | 2.5887 |
| **新词表（gpt-2 切分，含 `<\|endoftext\|>`）** | 2,603,786 | 4.027 | 7.13620 | 5.95% | **2.5565** |

最终组合比历史基线**好 5.52%**、比错配状态好 **4.27%**，且加载无警告、分词走快速路径。

**一个坑**：`gpt-2` 切分会把 `<|endoftext|>` 拆成 `<|`/`endoftext`/`|>`，重训词表后该字符串**不再是单个 token**，
demo 的分隔符切块因此失效（又退回 10 分钟的慢路径）。`train-tokenizer.py` 现在会**自动保留**
`<|endoftext|>` 为特殊 token；旧词表保留在 `assets/vocab-4k-word/`（`--split word` 复现）。

**副作用**：`pre` 规则从 `default` 改为 `gpt-2` 后语料编码略有变化（10 MB 切片 2,831,040 → 2,713,982 token，
约 −4%），现在的切分与词表训练时一致；早期模型是用旧规则训练出来的。

### 7.8 从零训练（预训练）而不是微调

前面所有例子都是**微调**：加载已经训练好的 `stories15M.gguf`，继续在别的语料上更新权重。**预训练**则是从**随机初始化**开始。演示程序只会加载 GGUF，所以"随机初始化"要构造一个随机权重的模型文件——
现在这件事由 [`tools/make-model.py`](../tools/make-model.py) 按配置生成（不再从官方权重派生，也不再下载模型）：

```powershell
cd vulkan-minitrain
$exe = ".\build\bin\Release\vulkan-minitrain.exe"

# 按 configs/minitrain-6l288-4k.json 生成随机权重模型（内嵌自训 4K 词表）
python tools\make-model.py --config configs\minitrain-6l288-4k.json
#  → assets/minitrain-6l288-4k-random.gguf（31.9 MiB，8.34 M 参数）
#     规则：所有 *_norm.weight 置 1.0，其余 N(0, 0.02)，同 seed 可逐字节复现

# 从零训练
& $exe --config configs\minitrain-6l288-4k.json --file .\assets\tinystories-10mb.raw --epochs 1 --lr 1e-6
```

**怎么确认它真的是"从零"**：随机初始化后的 loss 应当接近词表均匀分布的熵 —— 自训 4K 词表是 **ln(4096) ≈ 8.32**，
官方 32K 词表是 **ln(32000) ≈ 10.37**；如果加载后 loss 是 1–2，那说明你其实加载了训练好的模型
（`tools/check-random-init.py` 可以直接验）。

> 历史注记：早期版本用 `fetch-assets.py --random-model` 从官方 `stories15M.gguf` 派生随机权重
> （93.8 MB / 32K 词表），该选项已移除，相关实验数字是那个时期的。

实测（TinyStories 10 MB 切片 / 10,465 步 / Vulkan0）：

| 起点 | lr | train | **val** | acc | ppl |
|---|---|---|---|---|---|
| 随机初始化（**真预训练**） | **3e-5** | 5.80 | **5.405** | 0.16% | ~222 |
| 随机初始化 | 3e-4 | 12.19 | **13.29** | 0.01% | ❌ **发散** |
| 预训练权重（微调，作对照） | 1e-6 | 1.97 | **1.948** | 0.54% | ~7 |

| 399 KB 语料 / 373 步的 lr 扫描（随机初始化） | train | val |
|---|---|---|
| lr 1e-5 | 9.25 | 8.32 |
| **lr 3e-5** | 8.40 | **7.50** |
| lr 1e-4 | 8.18 | 7.98（训练均值更低但验证更差 → 过冲） |

三个必须知道的点：

1. **学习率比微调大 10–100 倍**（3e-5 vs 1e-6），但**远小于直觉**：3e-4 在 1 万步上就发散了（而在 34 步的小探针上看不出来）。
2. **没有 lr 衰减 / warmup / weight decay 时，长跑必然出问题**。`lr_opt::get_lr()` 默认 `lr_min = -1` → 学习率恒定；`wd = 0`。演示现在默认就是**步级 cosine 衰减 + warmup**（`--decay / --warmup / --lr-min / --wd`，见本章末「学习率调度（步级）」），旧的 epoch 级选项 `--decay-epochs / --lr-scale` 仅保留兼容。
3. **10 MB 语料训不出可用的模型**：从零一轮只到 ppl ~222（"刚开始学"），而微调一轮就是 ppl ~7。TinyStories 论文用的是**数亿 token**（1.79 GiB 全量 × 多轮）才让 1M–33M 模型说人话。
4. **全量语料在这台机器上跑不完一轮**：按实测吞吐外推，1.79 GiB（≈4.58 亿 token）→ **1,789,060 个 datapoint/轮 ≈ 24.5 小时/轮**，且数据集本身要 **7.3 GB 内存**（每轮实际处理 `ndata × 512` 个 token = 语料的 2 倍，因为窗口半步重叠）。

| 语料 | token | datapoints/轮 | 每轮墙钟（Vulkan0） | 数据集内存 |
|---|---|---|---|---|
| 10 MB 切片 | 2.82 M | 11,016 | **9 分钟** | 45 MB |
| 40 MB 切片 | 11.3 M | 44,071 | 36 分钟 | 180 MB |
| **1.79 GiB 全量** | 458 M | 1,789,060 | **≈ 24.5 小时** | **7.3 GB** |

> 因此这条路径的正确用法：**验证训练栈**（能不能跑、梯度是否流动、吞吐多少、衰减是否生效）；要真的把 15M 模型训到"说人话"，需要 24.5 小时 × 多轮 + 更大显存/内存预算，属于离线任务，不是交互演示。

#### 学习率衰减与权重衰减（旧 epoch 级选项，已被步级调度取代）

上游已暴露 `-lr / -lr-min / -decay-epochs / -wd / -epochs / -opt`（`common/arg.cpp`）；只有 `scale_epoch`（每轮减半的指数因子）没有 CLI。所以演示在解析参数后**直接给 `params.lr` 赋值**，不改上游源码；`--lr-scale` 是本演示特有的参数：

```powershell
# 每轮学习率减半，最低 3e-6（衰减只发生在 epoch 边界）
& $exe --model .\assets\stories15M-random.gguf --file .\assets\tinystories-10mb.raw `
       --epochs 3 --lr 3e-5 --lr-min 3e-6 --lr-scale 1 --decay-epochs 10
```

衰减公式（`common/common.cpp`）：

```cpp
lr = lr_min <= 0           ? lr0 :
     epoch >= decay_epochs ? lr_min :
                             lr0 * pow(0.5f, epoch * scale_epoch);
```

- 默认 `lr_min = -1` → 学习率恒定（这正是此前长跑发散的原因）
- 上面这组参数给出 **`3e-5 → 1.5e-5 → 7.5e-6`**；加 `--verbose` 后日志会逐 step 打印 `epoch N lr=…`，可直接核对
- **衰减按 epoch 计算，不是按 step**，所以至少 2–3 轮才看得出效果
- `--wd` 透传上游 `-wd`（AdamW weight decay，默认 0）

**实测（随机初始化 + TinyStories 10 MB + 上面那组参数，3 轮 / 27 分钟）**：

| epoch | lr | train | **val** | acc | 每轮墙钟 |
|---|---|---|---|---|---|
| 0 | 3e-5 | 5.800 | 5.405 | 0.16% | 542 s |
| 1 | 1.5e-5 | 5.055 | 4.943 | 0.18% | 537 s |
| 2 | 7.5e-6 | 4.797 | **4.785** | 0.19% | 530 s |

- **单调下降、无发散**（对比：恒定 lr 3e-4 跑一轮就发散到 val 13.29）
- 每轮收益递减：−0.46 → −0.16 —— 这是"该加数据，而不是继续加轮数"的信号
- 3 轮合计 **27 分钟**，吞吐 9,891 → 10,111 tok/s（与微调完全一致）

**把每轮数据放大 4 倍（40 MB 切片）反而变差** —— 这是本条路径最重要的坑：

| epoch | lr | train | **val** | acc | 每轮墙钟 |
|---|---|---|---|---|---|
| 0 | 3e-5 | 5.917 | 5.995 | 0.09% | 2,143 s |
| 1 | 1.5e-5 | 6.773 | 6.400 | 0.06% | 2,136 s |
| 2 | 7.5e-6 | 6.528 | **6.533** | 0.05% | 2,134 s |

（3 轮合计 **107 分钟**，42,091 步/轮）

轮内轨迹说明它是"过冲"而不是"学不动"：

| epoch | 第 1k 步 | 第 10k 步 | 第 20k 步 | 轮末 | 轮内最低 |
|---|---|---|---|---|---|
| 0 | 6.789 | 6.151 | 5.918 | 5.917 | **5.882**（起点 10.435） |
| 1 | 6.004 | 6.156 | **6.896** | 6.773 | 5.804 |
| 2 | 6.412 | 6.377 | 6.391 | 6.528 | 6.235 |

> 完整轨迹（含 5k / 30k 采样点）：epoch 0 `6.789 / 6.373 / 6.151 / 5.918 / 5.883 / 5.917`；
> epoch 1 `6.004 / 6.102 / 6.156 / 6.896 / 6.866 / 6.773`；epoch 2 `6.412 / 6.423 / 6.377 / 6.391 / 6.499 / 6.528`。

- **epoch 0 学得很好**（随机初始化的 10.44 → 轮内最低 5.88）
- **epoch 1 开始往上漂**（峰值 6.9），epoch 2 稳在 6.4–6.5 → 之后再没回来

原因是**步数**：40 MB/轮 = 42,091 步，3 轮共 **126,273 次更新**，是 10 MB 那组（31,395）的 **4 倍**；同样的 lr 序列（3e-5 → 1.5e-5 → 7.5e-6）在 4 倍步数下就会过冲。这与微调实验里"1e-5 在 2 万步还行、6 万步就劣化"是同一条规律：**有效步长预算 ≈ lr × 总步数**。

**结论**：加数据必须同时降 lr（大约按步数比例，这里应把 `--lr` 降到 ~7.5e-6），或者用更狠的衰减（`--lr-scale 2` 每轮除 4）。而且 `lr_opt` **只在 epoch 边界衰减**，42k 步/轮时轮内没有任何衰减保护，过冲无处可躲 —— 这是该机制的硬限制。

**验证：同一份 40 MB 数据，只把 lr 降到 7.5e-6（1 轮，36 分钟）**

| 配置 | 步数 × 轮数 | train | **val** | 墙钟 |
|---|---|---|---|---|
| 40 MB，lr 3e-5 | 42,091 × 1 | 5.917 | 5.995 | 36 min |
| **40 MB，lr 7.5e-6** | 42,091 × 1 | 4.868 | **4.331** | 35 min |
| 40 MB，lr 3e-5 衰减 | 42,091 × 3 | 6.528 | 6.533 | 107 min |

> `acc`：0.09% / 0.25% / 0.05%（32k 词表 top-1，参考意义有限）。

**只改学习率（其余完全不变），val 从 5.995 → 4.331（−1.66 nat），acc 0.09% → 0.25%。** 这精确验证了"有效步长预算 ≈ lr × 总步数"，也说明此前 40 MB × 3 轮的失败纯粹是 lr 没重调。

补充一个更好用的经验法则 —— **累计 `Σ lr×step`（把衰减算进去）落在 0.3–0.5 最健康**：

| 配置 | Σ lr×step | val | 结果 |
|---|---|---|---|
| 10 MB，lr 3e-5 × 1 轮 | 0.31 | 5.405 | 好 |
| **40 MB，lr 7.5e-6 × 1 轮** | **0.32** | **4.331** | **最好**（同样预算，4 倍数据） |
| 10 MB，lr 3e-5 衰减 × 3 轮 | 0.55 | 4.785 | 好 |
| 40 MB，lr 3e-5 × 1 轮 | 1.26 | 5.995 | 开始过冲 |
| 40 MB，lr 3e-5 衰减 × 3 轮 | 2.21 | 6.533 | 明显劣化 |
| 40 MB，lr 3e-4 × 1 轮 | 12.6 | 13.29 | 发散 |

**所以正确用法是：先把 `Σ lr×step` 定在 0.3–0.5，然后把这份预算花在尽可能多的 token 上** —— 同样是 0.31 的预算，10 MB 只到 5.405，40 MB 到 4.331。

> **干净树首构建的已知问题（ExternalProject 顺序竞争）**：`vulkan-shaders-gen` 是 CMake
> `ExternalProject`，在 VS 生成器下它的 `install` 步骤可能抢在 `configure`/`build` 之前执行，报
> `cmake_install.cmake:39 (file)` 或 `missing CMakeCache.txt`。此时手工补两步再继续即可，之后增量构建都正常：
>
> ```powershell
> $gen = "build/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders-gen-prefix/src/vulkan-shaders-gen-build"
> cmake --build   $gen --config Release
> cmake --install $gen --config Release
> cmake --build build --config Release --target vulkan-minitrain -- /p:TrackFileAccess=false /p:BuildInParallel=false
> ```

## 8. 构建时的四个坑（原来固化在 `run-demo.ps1` 里，现已写进文档）

脚本被删掉后，这些经验必须留在文档里 —— 它们是"不查就一定会浪费时间"的那类：

| # | 坑 | 症状 | 处理 | 平台 |
|---|---|---|---|---|
| 1 | MSBuild 的 File Tracker 依赖命名管道 | 构建瞬失败且无诊断，或 `TRK0002` / `MSB6006 CL.exe exited with code 2` | 设 `$env:TrackFileAccess = 'false'`，并给 MSBuild `/p:TrackFileAccess=false` | **仅 Windows**（MSBuild） |
| 2 | MSBuild 默认**并行构建** solution 里的多个工程 | `vulkan-shaders-gen` 外部工程的 install 抢在 build 之前，报 `file INSTALL cannot find ... : File exists`（误导性极强，且常伴随"日志里只有 MSBuild 横幅"） | 加 `/p:BuildInParallel=false`；**注意不是 `/m` 的问题**，加了 `/m` 反而更容易触发 | **仅 Windows**（MSBuild） |
| 3 | PowerShell 里原生命令的 stderr 会变成 ErrorRecord | 配 `$ErrorActionPreference='Stop'` 时，一个 warning 就误中断脚本；更隐蔽的是：**原生 stdout 混进返回值**，导致 `$code -ne 0` 永远为真（曾真实踩到，误判构建总是失败） | 把原生命令的输出 pipe 到 `Write-Host`，函数只返回 `$LASTEXITCODE` | **仅 Windows PowerShell**（脚本层面；Linux 的 bash + `set -e` 没这问题） |
| 4 | 运行时依赖当前工作目录 | 在别处运行报 `[demo] missing input:`（默认参数是 `assets/...` 相对路径） | 运行时 `cd` 到 `vulkan-minitrain/`，或给 `--model/--file` 绝对路径 | **跨平台通用** |
| 5 | CMake `ExternalProject`（`vulkan-shaders-gen`）在 VS 生成器下步骤乱序 | 干净树首构建报 `cmake_install.cmake:39 (file)` 或 `missing CMakeCache.txt`，错误指向 `vulkan-shaders-gen-install.rule` / `-build.rule`；重跑一遍不一定好 | 等 configure 完成后，手工 `cmake --build` + `cmake --install` 那个 prefix 的 build 目录，再重跑主构建；之后增量构建正常。注释上游 `BUILD_ALWAYS TRUE` 与 `/p:BuildInParallel=false` 都无效 | **VS 生成器** |

另外两个相邻的 MSBuild 怪癖（不属"四个坑"，但排查时会撞上）：

- `/nodeReuse:false` 在**内联参数**写法下会报 `MSBUILD : error MSB1001: 未知开关`，改用数组参数形式（`& $msb @args`）或干脆不写、改用环境变量 `MSBUILDDISABLENODEREUSE=1`。
- 该 MSBuild 版本**不接受 `/nologo`**（同样报 MSB1001）。

> 结论：换成 Linux 后，坑 1–3 集体消失（没有 MSBuild、没有 PowerShell），**只剩坑 4**，而它已经在演示程序里用"输入检查 + 明确报错"兜住了。

## 9. 重新编译（改代码或改源码树之后）

```powershell
cd vulkan-minitrain

# 增量构建（推荐）：MSBuild 会发现变更并重编依赖
cmake --build build --config Release --target vulkan-minitrain -- `
      /p:TrackFileAccess=false /p:BuildInParallel=false

# 只改演示自己的代码时，跳过项目依赖（约 4 秒；注意此路不会重编 llama/ggml）
msbuild build\vulkan-minitrain.vcxproj /p:Configuration=Release /p:Platform=x64 /p:BuildProjectReferences=false
```

- 完整重建（含 1566 个 Vulkan 着色器变体）约 **8–10 分钟**；改源码树里的 `ggml.c` 后必须走这条路才会生效。
- 从零配置：`cmake -S . -B build -G "Visual Studio 18 2026" -A x64`（可选 `-DLLAMA_TRAIN_DIR=<你自己的 llama.cpp 路径>`、`-DGGML_CUDA=ON`）。
- Linux 等价命令：`cmake --build build`（单配置，不需要 `--config`，也不需要那串 `/p:` 开关）。

---

## 相关文档

| 文档 | 内容 |
|---|---|
| [MODEL.md](MODEL.md) | stories15M 模型档案：结构 / 参数量 / TinyStories 背景 / 学习率实测 / 注意点 |
| [APPLY-PATCH.md](APPLY-PATCH.md) | 给源码树打 `ggml_cont` 补丁（拿到 1.86× 的 Vulkan 提升） |
| [PLACEMENT-AND-PERF.md](PLACEMENT-AND-PERF.md) | 落点与性能图文说明 |
| [../README.md](../README.md) | 项目总览、构建方法、实现要点 |

### 学习率调度（步级）

上游 `lr_opt` **只在 epoch 边界衰减**，所以一个很长的 epoch 内部完全没有保护 —— 我们之前在
「40 MB × 3 轮、lr 3e-5」上就见过明显劣化（val 6.533）。本 demo 自己实现 `get_opt_pars` 回调，
在**每个 optimizer step** 上算学习率：

```
lr = lr_min + (lr0 - lr_min) * 0.5 * (1 + cos(pi * t))     # t = (step - warmup) / (total - warmup)
```

| 选项 | 默认 | 说明 |
|---|---|---|
| `--warmup <n\|N%>` | 自动（1%，且不超过总步数的 1/10） | 线性 warmup 到 `lr` |
| `--decay <off\|linear\|cosine>` | `cosine` | 衰减曲线；`off` = 恒定 lr（旧行为） |
| `--lr-min <f>` | `0.1 × lr` | 衰减下限 |
| `--wd <f>` | 0 | AdamW weight decay |

**注意**：`--decay-epochs`、`--lr-scale` 这两个 epoch 级选项现在被忽略（保留只为兼容旧命令）。

**验证方法**：`MINITRAIN_LR_TRACE=1` 会把每一步的 lr 打到 stderr：

```
[lr] step 0/16  lr 0.0001
[lr] step 5/16  lr 8.51109e-05
[lr] step 15/16 lr 1.09834e-05
```

**预算经验**：由于 cosine 的平均 lr 约为 `0.5 × lr0`，总预算 `Σ lr×step ≈ 0.5 × lr0 × steps` ——
定 lr 时按这个算，落在 0.3–0.5 就是健康区间。

### 推理生成（`--infer`）要点

- 采样链顺序：`penalties → top_k → top_p → temp → dist`（惩罚必须在截断之前，与 llama.cpp 建议一致）。
- `--repeat-penalty`（默认 1.1）与 `--repeat-last-n`（默认 64）针对小模型常见的循环重复；实测同一 prompt/seed：
  penalty 1.0 与 1.1 的重复片段都要出现 3 次，**1.25 降到 1 次**。
- **结束符处理**：训练出的模型常把 `<|endoftext|>` 拆成 `<` `|` `end` `of` `text` `|` `>` 等多个 token 吐出来，
  而 `llama_vocab_is_eog()` 未必命中。程序会**缓冲以 `<` 开头的片段**，一旦拼成 `<|...|>` 就直接结束且**不打印**；
  不是控制符的文本会立刻刷出。实测 6 个 seed 均无 `<|endoftext|>` 泄漏，并在 62–185 token 处自然收尾。
- `MINITRAIN_TOK_TRACE=1` 可逐步打印 token id 与片段，用来排查结束符/乱码问题。

### 按配置建模型的流程与两个坑

1. **词表**：`python tools/train-tokenizer.py --corpus <语料> --out-dir assets/vocab-4k --vocab-size 4096`
   在完整 TinyStories（1.8 GB）上约 **9 分钟**（流式统计词频 + 3,837 次 BPE 合并），产出 `vocab.json` + `merges.txt`。
2. **模型**：`python tools/make-model.py --config configs/minitrain-6l288-4k.json`，默认输出
   `assets/<配置名>-random.gguf`。权重：所有 `*_norm.weight` = 1.0，其余 `N(0, init_std)`。
3. **训练**：`--config configs/xxx.json`（隐含 `--pretrain`）。首轮 loss 应 ≈ `ln(vocab_size)`。

**坑 1（GGUF 格式）**：tensor info 里的 offset 是**相对数据区起点**的，不是文件绝对偏移。
写绝对偏移时 `gguf_init_from_file()` 直接返回空，报错只有一句 `failed to load model from …`。

**坑 2（不要从程序里 shell 调用 Python）**：demo 曾经用 `system("python tools/make-model.py …")` 自动构建，
结果子进程继承了 `PYTHONHOME`/`PYTHONPATH` 污染，报 `Fatal Python error: Failed to import encodings module`。
现在 demo 只推导路径并打印命令，构建步骤显式执行。

**改词表大小**：重新跑第 1 步（换 `--vocab-size` 和 `--out-dir`），再改配置里的 `vocab_dir`，然后重跑第 2 步。
`n_embd` 等架构参数与词表大小互相独立，可以自由组合。

### 原始权重的归档（`backup/`）

官方检查点 `stories15M.gguf` 及其派生的 `stories15M-random.gguf`（32,000 词表）已移到 **`backup/`**：

- 它们会让"从零训练"悄悄变成"微调"，且词表与我们自训的 4,096 不一致（教训见报告 §6.11）；
- `backup/` 在 `.gitignore` 里（约 188 MB 不提交），其 `README.md` 说明来源与恢复方法；
- 现在默认入口是 `--config`：缺模型时程序会打印
  `python tools/make-model.py --config configs/minitrain-6l288-4k.json` 与后续训练命令；
- 仍想微调官方权重：`python fetch-assets.py` 会把它下载回 `assets/`，或从 `backup/` 移回。

### 校验模型是否为随机初始化

```powershell
python tools\check-random-init.py --model assets\minitrain-6l288-4k-random.gguf --config configs\minitrain-6l288-4k.json
```

判据（`minitrain-6l288-4k-random.gguf` 实测）：

| 检查项 | 随机初始化 | 本文件 | 官方检查点 | 我们训过的模型 |
|---|---|---|---|---|
| `*_norm.weight` | 恰好 1.0 | **1.00000（std 0）** | 0.758 / 0.958 / 1.193 | 0.242 / 0.549 / 0.618 |
| 其他张量 std | ≈ `init_std`（0.02） | **0.0200** | 0.061 | 0.448 |
| 嵌入行 \|cos\| | ≈ 0 | **0.0465** | — | — |
| 同种子重建 | 逐字节相同 | **True** | — | — |
| 换种子 | 不同 | **True** | — | — |

**顺带修掉一个格式问题**：GGUF 没有"张量信息之后的 alignment 字段"，对齐来自元数据 `general.alignment`
（默认 32）。`make-model.py` 早期多写了 4 字节，因为恰好落在对齐填充里才没出错；现在按规范写元数据。

### 学习率预算：按模型重新标定，别套用旧区间

实测（4K 自建模型，10 MB × 3 轮 = 15,750 步）：

| 调度 | lr0 | Σ lr×step | val |
|---|---|---|---|
| 恒定 | 3e-5 | 0.47 | 4.931 |
| cosine | 6e-5 | 0.47 | 4.969 |
| **cosine** | **3e-5** | **0.24** | **4.836（最好）** |

- **形状**（恒定 vs cosine）在同预算下几乎无差别 —— 别把"改成 cosine"当性能优化；
- **预算**是主导项，但最优值**随模型/词表变化**：32K 模型约 0.3–0.5，4K 模型约 0.24；
- 换模型时先跑三组（`--decay off` 与 `--decay cosine` 两个 lr）标定，再决定长跑参数。

### `--stride`：窗口重叠与步数

训练集是把语料切成 `n_ctx` 长的滑动窗口，默认步长 `n_ctx/2`（**一半重叠**）—— 于是**每个 token 出现在两个窗口里**，
每轮步数（以及报告的 token 数）都是"唯一 token 数"的约两倍。

| `--stride` | 10 MB 语料步数/轮 | 单轮耗时 | 说明 |
|---|---|---|---|
| 512（默认，= n_ctx/2） | 5,250 | 210.6 s | 每个 token 见两次，历史结果都用这个 |
| **1024（= n_ctx）** | **2,624** | **110.0 s** | 不重叠，覆盖相同、步数减半 |

注意：步数减半意味着**同 lr 下总学习预算也减半**（`Σ lr×step`）。换 stride 时要同步调 lr —— 例如从
σ≈0.24 的标定值出发，步数减半就把 lr0 加倍。

**统计口径提醒**：日志里的 `(N tokens)` 是**窗口槽位**，不是唯一 token 数（约为其 2 倍）；估算时间要用槽位 ÷ 吞吐。

**分词口径的额外发现（2026-10-24）**：把 `<|endoftext|>` 当作单个 CONTROL token 插入（当前实现）比让它由
BPE 编成约 7 个普通 token（10-23 的实现）**略差**：5 片跑完 val 1.78994 vs 1.69225，per-byte 0.6861 vs 0.6645
（差 3.3%）。原因是每片 token 数从 111.3 M 降到 108.6 M，分隔符处的训练信号变少。当前版本按 1.78994 发布，
这一口径的影响留待后续用同一数据集做对照研究。
