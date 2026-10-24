# 本地改动说明（已合并进源码）

> 本项目**基于 [llama.cpp](https://github.com/ggml-org/llama.cpp)**，重点是**在 Vulkan 后端上做训练**。
> 为训练场景做的三处改动已**直接合并**进源码树 `src/llama_train/`，不再以补丁脚本形式外挂。

## 1. 改动在哪

| # | 文件 | 目的 | 效果 |
|---|---|---|---|
| 1 | `src/llama_train/ggml/src/ggml.c` | 给 `MUL_MAT` 反向上游的两处 `OUT_PROD` 补 `ggml_cont()` | `OUT_PROD` 不再因转置视图被 Vulkan 拒绝（落点 CPU 60.9% → 0%），20 轮微调 19.48 → 10.48 s |
| 2 | `src/llama_train/ggml/src/ggml-vulkan/vulkan-shaders/out_prod.comp` | 寄存器分块（每线程 4×4 输出块） | 该算子 GPU 时间 −18%，端到端 −11.5%，吞吐 9,944 → 11,130 tok/s |
| 3 | `tests/test-backend-ops.cpp`（**不在本树**） | 打开 `OUT_PROD` 的 `trans_b` 对拍用例 | 证明转置视图下 CPU/GPU 结果不一致；本树未包含 `tests/` |

完整 diff、理由与度量见 **[`src/llama_train/LOCAL-CHANGES.md`](../src/llama_train/LOCAL-CHANGES.md)**。

两处改动都**不影响数值**：训练 loss 与未改动时逐位一致（40 KB 语料 8.13686 / 8.28643；TinyStories 10 MB 1.96785 / 1.94835）。

## 2. 验证

```powershell
cd vulkan-minitrain
cmake --build build --config Release --target vulkan-minitrain
.\build\bin\Release\vulkan-minitrain.exe --file .\assets\train-small.raw --epochs 1 --lr 1e-5
# 期望：train 8.13686 / val 8.28643
```

## 3. 想做 A/B 对照（回到上游行为）

本树不保留 `.orig` 备份。要对比"没有这些改动"的行为，把 `LOCAL-CHANGES.md` 里的 diff 反向应用到一份干净的上游检出：

```powershell
git apply -R .\LOCAL-CHANGES.diff      # 或按 diff 手工反向修改这两处
cmake -S . -B build-plain -G "Visual Studio 18 2026" -A x64 -DLLAMA_TRAIN_DIR=D:\llama.cpp
```

配置时用 `-DLLAMA_TRAIN_DIR=<你的 llama.cpp>` 就能让演示构建在别的源码树上。

## 4. 如何跟上游同步

1. 取一份新的上游 llama.cpp（或 `git pull`）；
2. 按 `LOCAL-CHANGES.md` 的三段 diff 重新施加改动（前两处是训练路径必需，第三处只影响测试）；
3. 重新配置构建，跑第 2 节的验证，数值应与本文档记录一致。

> 上游建议（可选）：第 1、3 条是可提交给 llama.cpp 的通用修复（反向后向的连续性约束、被默认关闭的对拍用例）；第 2 条是针对 `out_prod` 着色器的纯优化，可单独提 PR。
