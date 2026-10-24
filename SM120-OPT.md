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

## Machine assignment (2026-10-24)

- **.158 (8x RTX 6000D 84G)** — k7 recipe line (upstream-capable work:
  formal k7 validation, acceptance metrics, agent tiers). venv
  `/root/v60762` (dev440 stock), weights `/data/models/DeepSeek-V4.1-Flash-1017`.
- **.162 (8x RTX 6000D 84G)** — THIS fork's kernel line. Ops dir
  `/root/sm120opt_ops/`, venv `/root/sm120opt` (dev440), source checkout
  `/root/sm120opt_src` (this branch), weights rsynced to
  `/root/DeepSeek-V4.1-Flash-1017`. Do not touch other users' dirs on
  this host (NV personnel share it).

## DeepInfer absorption plan (kernel-level, from the 2026-10-20/22 campaign)

Validated SM120-usable assets from the DeepInfer investigation
(`DeepInfer-1017-SM120-测试指引-20261020.md` §12.2):

1. **FlashInfer 0.7.2 `b12x_fused_moe` (SM120a CuTe-DSL)** — first fork
   experiment: wire `--moe-backend b12x` for DSV4.1 MXFP4 experts, A/B vs
   DEEPGEMM_MXFP4 (k7 baseline: c32 1000.4 tok/s on .158). Known gaps
   from the campaign: mxfp4 weight layout undocumented, B12xMoEWrapper
   NVFP4-only + no EP (may need fork-side fixes — that IS the work).
2. **DeepGEMM PR#318 sm120 kernels** (cherry-pick into vendored DeepGEMM
   + arch-12 dispatch): `sm120_fp8_paged_mqa_logits` (indexer), 
   `sm120_tf32_hc_prenorm_gemm` (hyper-connection), `sm120_fp8_einsum`
   (recipe coverage limited — only (1,128,128)).
3. Keep as reference (NOT absorbable): DeepInfer's torch fallbacks from
   the link-opening campaign (slow, quality-degraded); mega_moe/mega_mhc
   (SM100f-only, no SM120 kernels exist); DeepEP GIN paths (no RDMA here).
