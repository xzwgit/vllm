# DSV4.1 SM120 NVFP4 KV — 观察性功能（Experimental）

**状态：EXPERIMENTAL / OPT-IN。默认 KV 仍为 fp8。** 本目录是功能载体与证据，不改变任何默认行为。

## 它是什么

让 `--kv-cache-dtype nvfp4_ds_mla` 在 SM120（RTX PRO 6000/6000D，cc 12.0）可用：
走 FlashInfer #5197 的 DSv4.1 双缓存记录（528B fp8 SWA + 288B NVFP4 压缩）。
上游 vLLM 将该格式门在 SM100（FlashMLA 路径）；上游 FlashInfer main（截至 2026-10-28）
不含 interleaved scale 读取能力——两处都需要补丁。

## 实测（2026-10-28，8×RTX 6000D 84G，vLLM 0.32.1rc1.dev463 + FI 0.7.2.post1，k7prob TP8）

| 项 | fp8 | nvfp4_ds_mla |
|---|---|---|
| KV 池 | 8,529,675 tok（8.13x@1M） | **13,290,418 tok（12.67x@1M，+55.8%）** |
| 吞吐（c1/c8/c32/64K-1K-c8） | 128.9/533.9/948.6/165.1 | 131.7/439.7/983.1/162.4（持平） |
| 确定性 | distinct=1 | distinct=1（补丁机械正确） |
| FP8 路径回归 | — | 无（footer 逐位一致实测） |
| 质量 | 基线 | **有损**（12*12 一题确定性答错；4-bit 压缩固有代价） |

## 何时启用

仅当 KV 容量是瓶颈（1M 上下文满并发、4 卡长窗场景）。质量敏感场景保持 fp8。

## 使用（两步，均幂等）

```bash
# 1. vLLM 侧：SM120 走 V4.1 记录 + dtype 门（3 文件）
python patch_vllm_nvfp4_sm120_upstream.py --vllm-dir <vllm>

# 2. FlashInfer 侧：interleaved scale 读取（19 文件 + nvcc AOT 编译）
python patch_flashinfer_dsv41_interleaved.py --flashinfer-dir <flashinfer> --build
#   产物 sparse_mla_sm120.so 装入 flashinfer/data/aot/（先于 jit-cache 加载）

# 3. 启动（其余参数任意，与 fp8 完全同配方）
vllm serve ... --kv-cache-dtype nvfp4_ds_mla
```

顺序坑：必须先打 FlashInfer 补丁——vLLM 侧能力探针检测 `kv_scale_placement` 参数，
没有它 nvfp4_ds_mla 会被安全拒绝（而不是读出乱码）。

## 出处与致谢

- vLLM 侧补丁：kacper-daftcode/vLLM-Moet `tools/dsv41_sm120/nvfp4_kv/patch_vllm_nvfp4_sm120_upstream.py`（其上游 PR 分支 dsv41-sm120-nvfp4-flashinfer）
- FlashInfer 侧补丁：同仓 `tools/dsv41_sm120/flashinfer_interleaved/{dsv41-interleaved.patch, patch_flashinfer_dsv41_interleaved.py}`
- 完整证据：llm-test `sm120opt-162/nvfp4kv-157/`（commit 164a1c1da）+ 工作区 data/nvfp4kv-157/RESULTS.md

## 回退

两补丁均幂等可逆（删除对应 marker 段或重装 flashinfer 包即恢复 footer-only 行为）。
