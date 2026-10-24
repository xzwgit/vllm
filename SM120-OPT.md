# SM120-OPT — RTX PRO 6000 / 6000D (SM120) serving optimization line

Decoupled optimization branch for DeepSeek-V4.1-Flash serving on 8x SM120
workstation GPUs (RTX PRO 6000 96G / RTX 6000D 84G). This line tracks
upstream `main` **opportunistically** (rebase when we choose, not every
drop) and is free to carry patches that upstream will not take yet.
Upstream-candidate fixes still go through separate PR branches
(`fix/dsv41-sm120-uniform-64`, `fix/mxfp4-cutlass-moe-sm120-guard`, ...);
nothing here is PR-bound.

## Baseline (2026-10-24)

- vLLM `0.32.1rc1.dev440` + FlashInfer `0.7.2` wheel, stock, TP8+EP8,
  DSpark, PIECEWISE graphs, util 0.90, capture=128*(k+1).
- Unified-matrix reference (vLLM 0.32.0 + dual patch tree, .158):
  k3/k4/k5 full 87-tier matrix; c32 1Kx1K k4 ~879-919 tok/s; single-stream
  k5 ~137 tok/s prose; KV pool 12.66M/12.07M/11.47M tokens (k3/k4/k5).

## Roadmap

1. **DSpark k=6/7 + probabilistic draft + block rejection** — code/agent
   single-stream lever (reference: sybil 4x96G TP4 build reports 557-573
   tok/s code single-stream at 5.4-5.6 accepted tokens/step, k7).
2. **indexer_kv_dtype=mxfp4 + indexer_sparse_logits=true** — official
   Blackwell recipe knobs, untested by us; saves indexer KV + possible
   speedup. Quality-gate before adopting.
3. **KV tiering (OffloadingConnector, host RAM + NVMe tiers)** — capacity
   without VRAM cost; hosts here have 1.2T RAM.
4. **FP4 KV (`nvfp4_ds_mla`) on SM120** — port/evaluate the
   kacper-daftcode/vLLM-Moet r13 approach (Apache-2.0): KV capacity x2 vs
   fp8; lossy (top-1 agreement ~0.974 incl. FP4 KV in their gate). Kernel
   level: needs their sm_120 steps or fresh port + DeepGEMM rebuild.
5. **b12x MoE backend evaluation** once upstream matures (oracle already
   lists B12X_MXFP4_MXFP8; Moet runs it fused on sm_120).

## Provenance / references

- sybil-solutions/local-ai-registry launch
  `vllm-moet-deepseek-v4.1-flash-1017-512k-tp4.json` (4x96G TP4 recipe,
  measured numbers + quality gate) and local-ai-images `vllm-moet-dsv41`.
- kacper-daftcode/vLLM-Moet (r13, sm_120 steps 1-28).
- Our validation data: llm-test repo campaign archives (m86, fi6161-b300,
  fi5898-repro).

## Ground rules

- Every optimization lands with: A/B on the same host+weights, engine
  health check per tier, quality probes (math/zh/tool-call + teacher-
  forced agreement for lossy items), raw logs with BOOT CMD.
- Lossy switches must be individually measurable and default-off.
