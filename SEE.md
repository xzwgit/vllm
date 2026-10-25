# SEE — SM120 serving optimization line (RTX PRO 6000 / 6000D)

> Branch renamed sm120-opt → **see** (2026-10-24). Same line, same rules.

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

## Artifact workflow (binding, 2026-10-24)

Every experiment on the .162 line: **test → pull artifacts to the local
workspace (`D:\llmtest\data\<campaign>\`) → push to GitHub for backup**
(`xzwgit/llm-test`, master, machine-keyed dir `sm120opt-162/` with README;
same-machine reruns land in subdirs + README section, never sibling dirs).
Archive rules: raw logs get a `=== BOOT COMMAND ===` header (api key
masked `***`) before upload; evidence (bench logs, probes, failure
signatures) travels with a one-line conclusion in the README.
## Experiment log — 2026-10-24 b12x first run (.162)

**Verdict: b12x on SM120 x DSV4.1-1017 is fast but numerically
untrustworthy.** TP8-only (oracle rejects EP), requires the #60762 port
(dev451 still asserts without it — third independent confirmation).
Throughput: c1 321.5 tok/s (+18.8% vs k7+DEEPGEMM 270.6), c32 1044.6
(+4.4% vs 1000.4). BUT exact-continuation tasks silently corrupt
(`32319=32319=` loops) while chat/tool/code pass — lighter sibling of the
flashinfer_cutlass corruption (vllm#63473). DEEPGEMM leg with identical
spec recipe is clean, so the damage is in the b12x kernel x DSV4.1 MXFP4
weight layout (undocumented per DeepInfer campaign). Next fork item:
audit the b12x weight/scale packing against FlashInfer's b12x tests.
Artifacts: llm-test `sm120opt-162/`.
## Experiment log — 2026-10-25 small hours (second night entry)

**Cross-machine validation of the dynamic-alignment patch** (.158 dev440):
8K c8 triple-repeat 663/666/642 vs stock 605.8 (+8.5%, stable); c32 within
its 902-1047 variance band, no regression. Combined with .162 dev451
(+12.4% c32 / +51.3% 8K c8): the patch never regresses; gain magnitude
grows with the vendored DeepGEMM version (dev451's BM64 tiles are more
efficient than dev440's). Evidence: llm-test b06703517.

**b12x direction locked**: FlashInfer's own SM120 b12x functional-API
accuracy suite passes 122/122 on this stack (incl. CUDA-graph replay and
micro-kernel paths) — the corruption is specific to vLLM's TPMoe-path
wiring for MXFP4xMXFP8. Next SEE item: route SM120 to the functional
`b12x_fused_moe` API (reference: kacper-daftcode/vLLM-Moet
tools/dsv41_sm120/b12x_moe_fused/, Apache-2.0).

**A-line (upstream knobs) closed**: indexer mxfp4 is explicitly gated to
sm_10x datacenter by vLLM (fails loud on SM120). KV tiering
(TieringOffloadingSpec, CPU 400GiB + fs tier) demonstrates real capacity
(32x512K live = 16.4M tokens > 12.66M GPU pool, all succeeded) at the
cost of -17% hot-path throughput → opt-in for capacity-starved shapes
only, not a default.
## Next-lever inventory (2026-10-25 patrol review)

Ranked by (benefit / difficulty), updated after the first-night results:

1. **DeepGEMM PR#318 sm120 kernel cherry-picks** (paged_mqa_logits /
   tf32_hc_prenorm / fp8_einsum into vendored DG + arch-12 dispatch) —
   the alignment patch proved the vendored-DG lever pays; PR318's
   kernels are the next concrete DG-level items. Medium difficulty
   (172-file divergence, pick per file), unknown-but-plausible gain.
2. **b12x fused-decode kernel** (kacper's .cu, Apache-2.0) — the
   c1 +18.8% is real (TPMoe throughput measured) but the only correct
   W4A8 route is a custom kernel; vendoring + adapting to the FI wheel
   API is a focused multi-day job. Park until #318 lands or machines
   free up.
3. **KV tiering as an opt-in recipe** — the capacity proof is done
   (512K×c32 live); the remaining work is documenting the
   hot-path −17% trade-off and shipping a deployment recipe, not code.
4. **use_thp:true engram tables** — untested lever from the m86 era;
   one boot + memory/perf probe on .158 when idle.
5. **Upstream the alignment patch** — once see has soaked it for a few
   days, file the same change against vllm main as a proper PR (it is
   a clean 3-file glue change, no fork-only APIs).
## Status snapshot 2026-10-25 AM (SEE-STATUS.md mirror)

All artifacts on llm-test (33fd16d60); THP probe incomplete (baseline
only, thp leg needs rerun); PR318 cherry-pick assessed NOT-FEASIBLE
(csrc/ compiled into .so, can't file-copy); next priority = upstream the
alignment patch as a vllm main PR. Full details: llm-test
`sm120opt-162/see-align-dyn/SEE-STATUS.md`.
## Rule change (2026-10-25, user decision)

**No further upstream PRs from this line.** SEE results stay on this fork
(exceptions already in flight: #63533 alignment patch was submitted before
this rule; it proceeds on its own). New findings land as commits to this
branch + artifacts to llm-test, per the artifact workflow.
