# 本树相对上游 llama.cpp 的本地改动

本目录是从上游 [llama.cpp](https://github.com/ggml-org/llama.cpp) 裁剪得到的副本，**改动已合并进源码**（不再用补丁脚本施加）。下面三处改动中，前两处在本树内生效；第三处针对 `tests/`，而本树裁掉了 `tests/`，仅作为记录与上游建议保留。

## 1. 改动清单

| # | 文件 | 目的 | 效果 |
|---|---|---|---|
| 1 | `ggml/src/ggml.c` | 给 `MUL_MAT` 反向上游的两处 `OUT_PROD` 补 `ggml_cont()` | `OUT_PROD` 不再因转置视图被 Vulkan 拒绝（落点 CPU 60.9% → 0%），20 轮微调 19.48 → 10.48 s |
| 2 | `ggml/src/ggml-vulkan/vulkan-shaders/out_prod.comp` | **重写为共享内存分块 GEMM**（64×64 tile / 256 线程 / 每线程 4×4 寄存器块 / 共享内存 padding）| `OUT_PROD` GPU 时间 29.17 → **11.63 ms（−60%）**，单步 GPU 36.41 → **18.79 ms（−48%）**，TinyStories 10 MB 单轮 538.8 → **431.6 s**，吞吐 9,944 → **12,413 tok/s** |
| 3 | `tests/test-backend-ops.cpp`（**不在本树**）| 打开 `OUT_PROD` 的 `trans_b` 对拍用例 | 用于证明转置视图下 CPU 与 GPU 结果不一致；本树未裁剪进 `tests/` |

> 前两处改动都不影响数值：训练 loss 与未改动时逐位一致（40 KB 语料 8.13686 / 8.28643；TinyStories 10 MB 1.96785 / 1.94835）。

## 2. 改动 1：`ggml.c` 的 `OUT_PROD` 连续性

```diff
--- a/ggml/src/ggml.c
+++ b/ggml/src/ggml.c
@@ -6925,10 +6925,15 @@
             if (src0_needs_grads) {
                 GGML_ASSERT(grad->ne[2] == src1->ne[2]);
                 GGML_ASSERT(grad->ne[3] == src1->ne[3]);
+                // Vulkan only accepts OUT_PROD with contiguous inputs, but backward passes
+                // often hand it transposed views. Only wrap when needed: ggml_cont() always
+                // inserts a CONT node, so wrapping unconditionally would copy every step.
+                struct ggml_tensor * op_a = ggml_is_contiguous(src1) ? src1 : ggml_cont(ctx, src1);
+                struct ggml_tensor * op_b = ggml_is_contiguous(grad) ? grad : ggml_cont(ctx, grad);
                 struct ggml_tensor * tmp =
                     ggml_out_prod(ctx, // [n,m,qq,rr]
-                        src1,          // [n,p,qq,rr]
-                        grad);         // [m,p,qq,rr]
+                        op_a,          // [n,p,qq,rr]
+                        op_b);         // [m,p,qq,rr]
                 if (!ggml_are_same_shape(tmp, src0)) {
                     GGML_ASSERT(tmp->ne[0] == src0->ne[0]);
                     GGML_ASSERT(tmp->ne[1] == src0->ne[1]);
@@ -6944,6 +6949,7 @@
                 ggml_add_or_set(ctx, cgraph, isrc0, tmp);
             }
             if (src1_needs_grads) {
+                struct ggml_tensor * grad_t = ggml_transpose(ctx, grad); // [p,m,qq,rr]
                 ggml_add_or_set(ctx, cgraph, isrc1,
                         // ggml_mul_mat(ctx,                   // [n,p,qq,rr]
                         //     ggml_cont(ctx,                  // [m,n,q1,r1]
@@ -6953,10 +6959,11 @@
                         // when src0 is bigger than tensor->grad (this is mostly the case in llama),
                         // avoid transpose of src0, rather transpose smaller tensor->grad
                         // and then use ggml_out_prod
+                        // ggml_transpose() is only a view, so materialise contiguous inputs
+                        // where the backend needs them (Vulkan rejects non-contiguous OUT_PROD)
                         ggml_out_prod(ctx,      // [n,p,qq,rr]
-                            src0,               // [n,m,q1,r1]
-                            ggml_transpose(ctx, // [p,m,qq,rr]
-                                grad)));        // [m,p,qq,rr]
+                            ggml_is_contiguous(src0)   ? src0   : ggml_cont(ctx, src0),   // [n,m,q1,r1]
+                            ggml_is_contiguous(grad_t) ? grad_t : ggml_cont(ctx, grad_t))); // [m,p,qq,rr]
             }
         } break;
         case GGML_OP_SCALE: {
```

## 3. 改动 2：`out_prod.comp` 重写为共享内存分块 GEMM

上游着色器每个线程算一个输出元素，内层循环每次迭代只有 1 个 FLOP 却要 2 次全局读（8 字节/FLOP），
实测约 **0.5 TFLOP/s**，占一个训练步 GPU 时间的 **80%**。

现在的实现（`ggml-vulkan/vulkan-shaders/out_prod.comp`）：

- 一个**工作组算一个 64×64 的 dst tile**：`TM=TN=64`、`TK=16`、256 线程、每线程 4×4 寄存器块
- A/B 的 k-strip **协作载入共享内存**（每线程 4 个元素，沿 i/j 连续 → 合并访问）
- 共享内存**行步长 +1 填充**（`[TM][TK+1]`）：不填充时 warp 内 16 个线程落在同一个 bank（步长 64 float ≡ 0 mod 32），
  内层循环被 16 路冲突串行化——这是本次实验里最大的坑
- 边缘用**绝对下标**判断（`gi0 + w < ne20`）：早前用 `ne20 - gi0` 会让越界线程的无符号减法下溢成巨大值，
  守卫失效、越界写内存（数值直接算错）
- 输出 tile 用 grid-stride 循环覆盖全部 tile，所以调度里只需 `min(ntiles, maxComputeWorkGroupCount[0])` 个工作组
- 后端配套改动：`ggml-vulkan.cpp` 的 `ggml_vk_op_f32` 里给 `GGML_OP_OUT_PROD` 单独算了 elements
  （该 pipeline 的 `wg_denoms[0] == 256`，故 `elements = 工作组数 × 256`）

**实测演进**（同机 RTX 4060 Ti，单步 = 一次训练迭代的 GPU 时间）

| 版本 | OUT_PROD | 单步 GPU 合计 | 398 KB 单轮 | TinyStories 10 MB 单轮 | 吞吐 |
|---|---|---|---|---|---|
| 上游（1 线程/输出） | 29.17 ms | 36.41 ms | 26.1 s | 538.8 s | 9,944 tok/s |
| 寄存器分块 4×4 | 23.80 ms | 31.04 ms | 23.7 s | 476.5 s | 11,245 tok/s |
| **共享内存 64×64（含 padding）** | **11.63 ms** | **18.79 ms** | **22.45 s** | **431.6 s** | **12,413 tok/s** |

数值三版完全一致（40 KB 8.13686/8.28643；10 MB 1.96785/1.94835）。

**副产品结论**：out_prod 从 29.17 降到 11.63 ms 后，单步墙钟里 CPU 侧（每步建图/调度）已经占约一半——
进一步提速的空间从 GPU kernel 转到了主机侧。

## 4. 改动 3：`test-backend-ops.cpp` 的 `trans_b` 用例（不在本树）

```diff
--- a/tests/test-backend-ops.cpp
+++ b/tests/test-backend-ops.cpp
@@ -11117,6 +11117,13 @@
                                                   256, 16, 16, {1, 1}, {nr2, 1}));
     }
 
+    // non-contiguous operand: the backward pass feeds OUT_PROD transposed views
+    // (ggml.c, ggml_compute_backward, GGML_OP_MUL_MAT), so cover that layout here.
+    for (int64_t ne2 : {1, 8}) {
+        test_cases.emplace_back(new test_out_prod(GGML_TYPE_F32, GGML_TYPE_F32,
+                                                  256, 16, 16, {ne2, 1}, {1, 1}, /*trans_b = */ true));
+    }
+
     // add_id
     for (ggml_type type_a : {GGML_TYPE_F32}) {
         for (ggml_type type_b : {GGML_TYPE_F32}) {
```

## 5. 验证方式

```powershell
# 数值：与未改动时逐位一致
cmake --build build --config Release --target vulkan-minitrain
.\build\bin\Release\vulkan-minitrain.exe --file .\assets\train-small.raw --epochs 1 --lr 1e-5
# 期望 train 8.13686 / val 8.28643
```

如果要回到上游行为做 A/B 对照，把本节的两处 diff 反向应用即可（`patch -R` 或 `git apply -R`，本树不保留 `.orig` 备份）。
