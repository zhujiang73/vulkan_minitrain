# 立项：训练图 static 化（摊掉每步建图开销）

**状态**：已立项，未开工
**日期**：2026-10-20
**前置结论**：见 [2026-10-19-PRETRAIN-REPORT.md](2026-10-19-PRETRAIN-REPORT.md) §6.7、§6.8

---

## 1. 问题与目标

### 现状

训练每步（一个 datapoint）在主机侧要做：

```
llama_context::opt_epoch_iter()
  ├─ memory->init_batch(...) / mctx->apply()
  ├─ res->reset()
  ├─ model.build_graph(gparams)          ← 重建整张训练图（约 3,122 节点）
  ├─ ggml_init(meta context)             ← 4*size_gf 个 tensor 的元数据
  ├─ ggml_opt_prepare_alloc(...)
  ├─ ggml_opt_alloc(...)                 ← 非 static：内部还会 ggml_opt_build() 追加 loss/反向节点
  ├─ res->set_inputs(&ubatch)            ← 512 个 token 拷进输入张量
  ├─ 写 labels
  ├─ ggml_opt_eval(...)                  ← 真正上 GPU
  └─ ggml_free(meta context)
```

**实测代价**（398 KB 语料、`--ctx 512`、稳态第 2 轮）：单轮 15.20 s / 393 步 = **38.7 ms/步**，
而同一批步的 **GPU 仅 18.79 ms/步** → **主机侧约 20 ms/步（52%）**。

### 目标

把"每步重建整张图"变成"**每个阶段只建一次**"：

| 指标 | 现状 | 目标 |
|---|---|---|
| 单步墙钟（398 KB） | 38.7 ms | **≤ 25 ms** |
| 稳态单轮（398 KB） | 15.20 s | **≤ 10.5 s** |
| 吞吐（398 KB） | 12,563 tok/s | **≥ 17,000 tok/s** |
| 数值 | — | **与现状逐位一致** |

---

## 2. 为什么不能"缓存现有图"（已实测，别重犯）

尝试过按 ubatch 形状缓存已建好的图，撞两个库级约束：

1. `ggml_opt_eval()` 结束会复位 `eval_ready` → 每步仍必须调 `ggml_opt_alloc()`；
2. llama.cpp 用的是**非 static 图模式**，`ggml_opt_alloc()` 里的 `ggml_opt_build()` 会**往图里追加
   loss/反向节点**，同一张图 build 第二次即
   `GGML_ASSERT(!(node->flags & GGML_TENSOR_FLAG_LOSS) && "support for extra loss terms not implemented")`。

**结论**：必须切到 ggml-opt 的 static 图模式，而不是在现有模式下缓存图。

---

## 3. static 模式的契约（读库确认）

`ggml_opt_init(params)`：

```cpp
result->static_graphs = result->ctx_compute;                 // 传了 ctx_compute 才算 static
result->gf = ggml_new_graph_custom(ctx_compute, ..., grads);  // 库自己从 outputs 建前向图
ggml_build_forward_expand(result->gf, result->outputs);
ggml_opt_build(result);                                       // 建 gb_grad / gb_opt / 优化器状态
```

要求：

- `ctx_compute`、`inputs`、`outputs` **三者都要给**；
- `ggml_opt_build()` 断言 `inputs->data` 非空 → **调用方必须先把图分配好**
  （`ggml_backend_sched_alloc_graph`）；
- `ggml_opt_build()` 只遍历一次 gf 收集 PARAM、断言无 LOSS 节点 → **只在 init 时做一次**；
- `ggml_opt_alloc()` 首次做 `dup_graph` + `sched_alloc_graph`，之后**同一 graph 指针直接早退**
  （只把 `eval_ready` 置 true）→ 每步 alloc 变成 O(1)；
- `ggml_opt_eval()` 在 static 模式下**不清空** gf/gb_grad/gb_opt/allocated_graph 指针 → 跨步存活；
- `ggml_opt_labels()` / `ggml_opt_inputs()` / `ggml_opt_outputs()` 在 static 模式下指针稳定
  （分配在 `ctx_static`）→ 每步只需重写数据。

---

## 4. 方案

### 4.1 建图与初始化（一次性）

在 `llama_context::opt_init()` 末尾（此时 `cparams.training`、`n_ubatch`、`n_ctx` 都已定）：

1. 造形状模板 ubatch：`n_tokens = n_ubatch`、单序列、token 用 0 填充（只为形状）；
2. `balloc->init(...)` → `memory->init_batch(*balloc, cparams.n_ubatch, true)` → `mctx->apply()`；
3. 在 **no_alloc 的 meta context**（`ctx_compute`）里构造图（等价于现在 `opt_epoch_iter` 里那段），
   拿到 `gf`、`inp_tokens`（inputs）、`logits`（outputs）；
4. `ggml_backend_sched_alloc_graph(sched, gf)` 让 `inputs->data` 有效；
5. `ggml_opt_params` 填 `ctx_compute/inputs/outputs/backend_sched/opt_period/get_opt_pars/optimizer`，
   调 `ggml_opt_init()`。

### 4.2 每步只更新输入

`opt_epoch_iter()` 在 static 模式下：

```
mctx = memory->init_batch(...); mctx->apply();
res->set_params(gparams);        // 刷新 mctx/ubatch 指针
res->set_inputs(&ubatch);        // 512 token → 输入张量
写 labels;
ggml_opt_alloc(opt_ctx, train);  // 早退
ggml_opt_eval(opt_ctx, result);
```

**不再** `res->reset()`、`model.build_graph()`、`ggml_init()`、`ggml_opt_prepare_alloc()`。

### 4.3 训练相 / 评估相

两相图结构不同（评估无梯度）。做法：**两套 static 图 + 两套 opt context**（各建一次），
或一套 context 在切相时重建一次（每轮 2 次，可接受）。**M2 决定**，先按"切相重建一次"实现，最省事。

### 4.4 回退开关

保留非 static 路径，用环境变量 `MINITRAIN_STATIC_GRAPH=0` 强制回退，便于 A/B 与线上兜底。

---

## 5. 里程碑

| 阶段 | 内容 | 验收 |
|---|---|---|
| **M0**（0.5 天） | 把 §3 的契约写成最小 spike：在 demo 里用固定 ubatch 直接调 `ggml_opt_init(static)`，跑 20 步 | loss 曲线与基线一致（20 步后偏差 < 1e-6） |
| **M1**（1 天） | `opt_init` 一次性建图 + static init；`opt_epoch_iter` 走 static 分支 | 40 KB 训练 **8.13686 / 8.28643** 逐位一致 |
| **M2**（0.5 天） | 训练相/评估相切换（各建一次）；多 epoch 复用 | 398 KB 两轮数值一致；单轮 ≤ 10.5 s |
| **M3**（0.5 天） | 边界与兜底：非标准 ubatch、`n_ubatch != n_ctx`、CPU 后端、`--no-save`、内存泄漏（`ggml_opt_free`） | 10 MB 一轮 **1.96785 / 1.94835**；`Task Manager` 无持续增长 |

---

## 6. 风险清单（按严重度）

| # | 风险 | 说明 | 缓解 |
|---|---|---|---|
| 1 | **图里绑定了首步 mctx 的指针** | 若训练图的 mask/KV 指针来自第一个 memory context，之后每步 mctx 都是新的 → 数据错位。**这是最大风险** | M0 先验证：static 跑 20 步看 loss 是否一致；若不一致，需要把 mctx 的生命周期与 static 图绑定（每 epoch 一个 mctx，或让 `set_params` 真正 rebind） |
| 2 | eval 相图结构不同 | 无梯度图不能复用训练图 | 切相重建一次（4.3） |
| 3 | `labels` 指针稳定性 | 非 static 下 labels 每步新建；static 下在 `ctx_static` 里 | 每步 `ggml_set_zero` + 写 1.0，不依赖指针变化 |
| 4 | 内存增长 | 每 epoch 重建两套图 | `ggml_opt_free` + meta ctx 释放；M3 用 5 轮观察 RSS |
| 5 | 与 `--ctx` / `--ubatch` 组合 | 形状变化需重建 | 形状签名不匹配时回退非 static 路径 |

---

## 7. 验收与回退

**数值**（必须逐位一致，任何一项不一致即视为失败）：

| 场景 | 期望 |
|---|---|
| 40 KB，1 轮，lr 1e-5 | train **8.13686** / val **8.28643** |
| 398 KB，1 轮，lr 1e-5 | train **7.83541** / val **7.63368** |
| TinyStories 10 MB，1 轮，lr 1e-6 | train **1.96785** / val **1.94835** |

**性能**：398 KB 稳态单轮 ≤ 10.5 s（现状 15.20 s）；单步 ≤ 25 ms（现状 38.7 ms）。

**回退**：`MINITRAIN_STATIC_GRAPH=0` 或删除本改动的 `llama-context.cpp/.h` 三个 hunk；
改动记录进 `src/llama_train/LOCAL-CHANGES.md` 的改动 4。

---

## 8. 预期收益的边界

即使 fully 摊掉建图，单步仍有 `mctx->init_batch/apply`、`set_inputs`（512 token 拷贝）、labels 写入、
`ggml_opt_eval` 的 scalar 读取等固定开销。按 §1 的实测分布，**乐观上限**是单步 18.8–22 ms
（≈ 1.75× 吞吐，13.4k → 23k tok/s 量级）；若 M1 只做到 28 ms（1.4×），也应视为成功 —— 再往下就要动
`opt_period`/梯度累积与真正的 batch>1 训练图了。

---

## 9. M0 结果（2026-10-21）：朴素做法失败，原因已定位

**做法**：首个 datapoint 建图 → `ggml_backend_sched_alloc_graph()` 让 `inputs->data` 有效 →
`ggml_opt_free()` 旧 context → 用 `ctx_compute/inputs/outputs` 重新 `ggml_opt_init()`（static）。

**现象**：

- 不崩，能跑完 33 步，但 **loss 恒为 10.375 = ln(32000)**（均匀分布，模型完全没在学）；
- epoch 末尾切换评估相时 **访问违例 0xC0000005**。

**根因**（读 `ggml-opt.cpp` 确认）：

1. static 模式下 `ggml_opt_alloc()` 首次调用会先 `ggml_backend_sched_reset()` —— **释放调度器给图分配的 buffer**，
   我们预分配的那批（含 `inp_tokens`）随之失效；
2. 紧接着 `dup_graph()` 通过 `map_tensor()` **把 data/buffer 指针原样拷进副本**（副本与原图共享 buffer）；
3. `sched_alloc_graph(copy)` 看到这些张量 `data != nullptr`，**跳过分配**；
4. 于是计算图读的是**已释放的内存** → 输入恒为 0 → logits 均匀（ln 32000）；epoch 末尾换相时再踩一次就崩。

**结论**：lla.cpp 的图张量（包括 `inp_tokens`）生命周期由 `ggml_backend_sched` 管理，与 ggml-opt
static 模式"输入必须常驻分配"的前提冲突。**要让它成立，输入张量必须落在调度器之外的常驻 buffer 里。**

**已回滚**：`llama-context.cpp/.h` 全部还原，`40 KB 8.13686/8.28643`、`398 KB 22.56 s` 复现，基线无损。

## 10. M1' 修订方案（下次从这里开始）

1. 为训练图**单独建一个常驻 buffer**：把 `inp_tokens`（以及需要每步写入的 pos/mask 输入）
   放进一个独立 `ggml_context`，用 `ggml_backend_alloc_ctx_tensors()` 直接分配，不经调度器；
2. 让 `dup_graph` 共享这批常驻指针（它本来就会共享 `data`），并在 `sched_reset` 之后仍然有效；
3. 其余中间张量仍交给调度器；
4. 再按 M0 的流程 init static、验证 40 KB 数值。

**验收**：40 KB `8.13686 / 8.28643` 逐位一致；否则回到 M0 分析。
**风险**：`res->set_inputs()` 内部可能对非调度器分配的输入做 `ggml_backend_tensor_set`（应该没问题，
它只要求 `data`/`buffer` 正确），但需要实测确认。

---

## 11. M1' 结果（2026-10-21）：整图常驻 buffer 撞到第二层约束

**做法**：把训练图的张量一次性分配到调度器之外的常驻 buffer
（`ggml_backend_alloc_ctx_tensors(res->get_ctx(), backend_ptrs[0])`），再按 M0 流程 init static。

**现象**：

```
llama-kv-cache.cpp:1493: GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer)) failed
```

**原因**：KV-cache 的索引输入（`set_input_k_idxs` / `set_input_v_idxs`）是**直接写 `dst->data`** 的，
因此这些张量必须在 **host buffer** 上；把整图放到设备 buffer 直接非法。

**由此得到的关键结论**：ggml-opt 的 static 模式要求"输入常驻分配"，而 llama.cpp 的图是**混合放置**的
（大部分在 GPU、少数输入必须在 host）。所以 static 化必须做**按张量放置感知的分配**，即：
对每个张量用 `ggml_backend_sched_get_tensor_backend()` 查它应属于哪个 backend，再按其 buft 分配。

## 12. M1'' 修订方案（下次从这里开始）

1. **补一个小 API**（ggml 侧，6 行，也可顺手提上游）：`ggml_graph_n_leafs()` 与 `ggml_graph_leaf(gf, i)`。
   目前只公开了 `ggml_graph_n_nodes()` / `ggml_graph_node()`，**leaf（输入张量基本都在 leaf 里）无法枚举**，
   这是本次卡住的直接原因。
2. 建好图后：遍历 `nodes + leafs`，对缺 `data` 的张量查
   `ggml_backend_sched_get_tensor_backend(sched, t)`（llama.cpp 已用 `set_tensor_backend(..., backend_cpu)`
   显式 pin 了 host 输入，见 `llama-graph.cpp:2988`），按 buft 分组成若干个常驻 buffer，
   用 `ggml_backend_tensor_alloc()` 逐个绑定。
3. 然后照 M0 流程 `ggml_opt_init(static)`；`sched_reset` 只会释放调度器自己的 buffer，不会碰我们的。
4. 验收：40 KB `8.13686 / 8.28643` 逐位一致。

**风险**：`get_tensor_backend()` 对未被显式 pin 的张量可能返回 null → 需要兜底（默认用主设备 buft）。
**已回滚**：M1' 改动全部还原，40 KB **8.13686 / 8.28643** 复现，基线无损。

---

## 13. M1'' 结果（2026-10-21）：第三次尝试，在库内部崩

**做法**（新的思路，绕开 leaf 访问器问题）：不再自己分配，而是
1. 让调度器照常 `ggml_backend_sched_alloc_graph(gf)` —— 放置（host/device）与分配都由它决定，天然满足 KV 索引的 host 要求；
2. 改 ggml-opt：static 模式下**跳过** `ggml_opt_alloc()` 里的 `ggml_backend_sched_reset()`，
   这样那批 buffer 不会被释放，`dup_graph` 共享的指针依然有效；
3. 再用 `ctx_compute/inputs/outputs` init static。

**现象**：第一步就 **访问违例 0xC0000005**，连 progress 行都没打印出来 —— 崩在 `ggml_opt_init`（static）内部，
很可能是 `ggml_opt_build` 在"没有先 `sched_reset`"的调度器状态上分配 `ctx_static` 张量导致的
（`ggml_backend_alloc_ctx_tensors(ctx_static, ggml_backend_sched_get_backend(sched, 0))`）。

**判断**：这个方向需要**调试器级别的排查**（WinDbg/VS 附加），而不是 7 分钟一轮的构建试验。
三次尝试分别是三层的语义约束：

| 尝试 | 卡点 | 层次 |
|---|---|---|
| M0 | `ggml_opt_build` 会改图，只能 build 一次 | ggml-opt API 语义 |
| M1 | `sched_reset` 释放预分配 buffer，dup 后悬空 | 调度器/图内存所有权 |
| M1' | KV 索引输入必须在 host buffer | 混合放置 |
| M1'' | 跳过 reset 后库内部崩溃 | 调度器状态一致性 |

**已回滚**（6 处，含 ggml-opt 还原），基线无损：40 KB **8.13686 / 8.28643**、398 KB **22.58 s**。

## 14. 建议：暂停此立项

理由：

1. **收益有限**：§1 估算的上限是 1.75×（12.5k → 22k tok/s 量级）；而 `--ctx 1024` 零成本拿到 −7%。
2. **成本已经明显**：三次尝试 ×（一轮改动 + 7.4 分钟构建 + 数值对拍）≈ 每轮 25–40 分钟，且失败点一层比一层深。
3. **方法不对**：这类"库内部所有权/状态"问题需要调试器，而不是构建-运行循环。

**如果将来重启**，建议顺序：
1. 先用 WinDbg 定位 M1'' 的崩溃点（`ggml_opt_init` 内）；
2. 或者换一条更彻底的路：**不经 ggml-opt**，自己写 6 行的 AdamW 更新（我们只需要 `ggml_opt_result` 的统计能力，那部分可以自己算），
   训练图与反向图完全自持 —— 这其实是本项目长期方向的自然延伸。
