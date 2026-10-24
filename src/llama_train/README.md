# llama_train

本目录是 [llama.cpp](https://github.com/ggml-org/llama.cpp) 的裁剪副本，作为 **vulkan-minitrain** 的训练栈。
相对上游的本地改动（已合并进源码）见 [LOCAL-CHANGES.md](LOCAL-CHANGES.md)。

裁剪范围：保留 `ggml/`（cpu / vulkan / cuda 后端）、`src/`、`common/`、`include/`、`vendor/`、`cmake/` 与
`LICENSE`、`licenses/`；移除 `tools/`、`tests/`、`examples/`、`docs/`、`models/`（vocab 与模板）、
转换脚本、CI/lint/agent 配置等。

许可证：MIT，见 [LICENSE](LICENSE)；vendored 依赖的许可证见 [licenses/](licenses/)。
