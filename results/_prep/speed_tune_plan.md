# Phase-5 SPEED-TUNE bench matrix — QUEST-30B-RL Q5_K_M on Intel Arc B70

DESIGN ONLY (the B70 is busy with a live perplexity run). Nothing here was executed.
Engineer runs this when the GPU frees up. Goal: push the single shipping quant
(Q5_K_M, 20.23 GB, fits the ~30 GiB B70 fully at `-ngl 99`) to its B70 optimum.

TARGET (must meet/beat): **pp512 1154.6 t/s, tg128 79.9 t/s** — the same-family A3B
reference (35B/256E Q5_K) on the MoE-reorder tree, `-ngl 99 -fa 1`, from `results/00_baseline.txt`.
QUEST is *smaller* (30B total, 128E/8A, no shared expert, ~3.3B active) so it should
**meet or exceed** the reference at equal flags; treat 1154.6/79.9 as the floor, not the ceiling.

## Shorthand (source env.sh first — pulls oneAPI setvars + all paths/flags)
```
source $HOME/quest-turbo/env.sh
M=$HOME/quest-turbo/artifacts/quest-30b-rl-Q5_K_M.gguf   # $ART/quest-30b-rl-Q5_K_M.gguf
MOE=$HOME/llama.cpp-sycl-moe-ready/build/bin              # $MOE_BIN == $RUN_BIN (speed tree)
MAIN=$HOME/llama.cpp/build/bin                           # $BIN (stock SYCL)
```
env.sh already exports the ship defaults: `GGML_SYCL_DISABLE_DNN=1`, `NGL=99`, `UBATCH=4096`,
`BBATCH=8192`. Commands below inherit `DISABLE_DNN=1` unless a speculative row overrides it.
Each `llama-bench` row already loads the model once (~20-25 s warm) and reorders the MoE
weights at load; prefer comma-lists in ONE invocation to amortize that load cost.

## Verified facts that shape the matrix (from source/build inspection, not memory)
- MoE tree HEAD `0b98ca1b4`. Carries, and confirmed in the binary:
  - `mul_mat_id` SYCL **reorder for Q4_K/Q5_K** (commit a7597d733) + Q6_K (6b856575a). Q5_K_M
    benefits. Auto-on via `GGML_SYCL_DISABLE_OPT=0` (default). Confirm `reorder: Y` in the load log.
  - **topk-moe router fusion** default-ON, +2.7% decode on the prior campaign (b311f6065). A/B with
    `GGML_SYCL_DISABLE_TOPK_MOE=1`. Commit note: "declines routing-bias graphs" — QUEST has
    no routing bias (qwen3_moe, no bias) so it SHOULD fire; verify the A/B shows a delta.
  - single-token expert-aggregate reduce (3195b5480) — helps tg.
  - gated-delta-net glue fusion (0b98ca1b4) — **N/A to QUEST** (standard GQA, not delta-net).
- Flash-attn IS compiled (head_dim-128 tile + vec kernels present in libggml-sycl). `fattn.cpp`
  `get_best_fattn_kernel`: head_dim 128 OK; requires `V->ne[0]==K->ne[0]` (128 OK); requires
  **`K->type == V->type`** (no FA_ALL_QUANTS) → `-ctk` and `-ctv` MUST match; q8_0 KV is FA-supported.
  Decode kernel split: f16 KV (gqa_ratio 8, mask, max_bias 0 → gqa_opt) → TILE kernel; q8_0 KV
  (Q->ne[1]<=2) → VEC kernel. So KV type changes the decode KERNEL, not just bandwidth — measure it.
- CPU: Ryzen 9 5950X, 16C/32T → bench default `-t 16` (math threads). At `-ngl 99` full offload
  the CPU touches only sampling/non-offloaded glue → `-t` expected ~flat (low-priority probe).
- VRAM math (Q5_K_M 20.23 GB model): f16 KV ≈ 96 KiB/token (2·4 KV-heads·128·48L·2B). At d=65536
  ≈ 6.0 GiB KV → ~26 GiB total, fits. d=131072 f16 ≈ 12 GiB → OOM; q8_0 halves it → fits.
  So deep-depth tests beyond 65536 are **q8_0-KV-only**.

## Proven-from-the prior campaign vs speculative (per knob)
| knob | status | why |
|---|---|---|
| MoE-reorder tree vs main | PROVEN | baseline 1154.6/79.9 vs 993/74.8 (+16% pp / +6.8% tg) |
| `-ub 4096` (un-starve MoE expert GEMMs) | PROVEN-the prior campaign | ship flag; biggest pp lever |
| `-fa 1` on | PROVEN-the prior campaign | ship flag; FA compiled & supports head_dim128 + q8_0 KV |
| KV `q8_0` (`-ctk/-ctv q8_0`) | PROVEN-the prior campaign ship choice; tg delta = the measurement | switches decode to VEC kernel + halves KV BW at depth |
| topk-moe router fusion | PROVEN-the prior campaign (+2.7% dec) — A/B to confirm it fires on 128E/no-bias | default on |
| `-b` (logical batch) | SPECULATIVE-low | bounded by ub; flat for pp512/pp2048 once `-b>=ub` |
| `-t` thread count | SPECULATIVE-low | flat at full offload |
| `GGML_SYCL_DISABLE_DNN=0` (oneDNN GEMM for prefill) | SPECULATIVE | may raise pp; ship keeps =1 for deterministic decode |
| `GGML_SYCL_DISABLE_GRAPH=0` (SYCL graph) | SPECULATIVE-low | decode is exec-bound not submit-bound (per b311f6065) → likely inert |
| `GGML_SYCL_ENABLE_FUSION=1` (rms_norm+mul epilogue) | SPECULATIVE-low | measured inert on the prior campaign (-0.3%); skip unless time |

## ORDERING (find the win fastest under the 2-hour clock)
Run in this order so the highest-value data lands first if interrupted. Times are rough.
1. **B0 anchor** (ship config, MoE tree) — locks the reference, confirms QUEST ≥ target. ~1.5 min
2. **B1 `-ub` sweep (pp only)** — the dominant pp512 lever; finds prefill optimum (4096 vs 2048 for 128E). ~3 min
3. **B2 FA on/off** — confirms FA help/neutral; gives tg128 anchor at f16 KV. ~2 min
4. **B3 KV q8_0 @ d=0 (tg)** — q8_0-KV decode cost vs B0 f16 (VEC vs TILE kernel). ~2 min
5. **B5 build A/B (main vs MoE)** — confirms the +16%/+6.8% MoE-tree win transfers to QUEST. ~3 min
6. **B4 depth + KV type** — the agentic payoff (QUEST runs long ctx); shows q8_0-KV tg win at depth. ~12 min
7. **B6/B7/B8 speculative env** — DNN=0 (pp), topk-moe A/B (dec), SYCL graph (dec). ~6 min
8. **B9/B10 low-priority** — `-t` and `-b` controls (expect flat). ~4 min
Total ~35 min of GPU time; well inside 2 h. If the clock is tight, steps 1-5 alone decide the ship flags.

## BENCH MATRIX (concrete invocations)

### B0 — anchor / ship config (PROVEN). f16 KV to match the 1154.6/79.9 baseline exactly.
```
$RUN_BIN/llama-bench -m $M -ngl 99 -fa 1 -ub 4096 -b 8192 -p 512 -n 128 -r 3
```
Pass if pp512 >= 1154.6 and tg128 >= 79.9. This is the number we report as the headline.

### B1 — `-ub` sweep, prefill only (PROVEN-the prior campaign; biggest pp lever)
```
$RUN_BIN/llama-bench -m $M -ngl 99 -fa 1 -b 8192 -ub 512,2048,4096 -p 512,2048 -n 0 -r 3
```
Expect pp to climb with ub (512<<2048<=4096). If pp2048@ub2048 ~ ub4096, 2048 may be the 128E
sweet spot (fewer experts than the 256E ref → smaller expert GEMMs may saturate sooner). Pick max-pp ub.

### B2 — flash-attn on/off (PROVEN-the prior campaign ship=on; quantify the delta)
```
$RUN_BIN/llama-bench -m $M -ngl 99 -ub 4096 -b 8192 -fa 0,1 -p 512 -n 128 -r 3
```
Expect fa1 >= fa0 on tg (and required for q8_0 KV). If fa0 wins pp at d=0, still keep fa1 — the
agentic win is at depth (B4), and KV q8_0 needs FA.

### B3 — KV cache type @ d=0, decode cost (PROVEN-the prior campaign ship=q8_0; delta is the measurement)
```
$RUN_BIN/llama-bench -m $M -ngl 99 -fa 1 -ub 4096 -b 8192 -ctk q8_0 -ctv q8_0 -p 0 -n 128 -r 3
```
Compare tg128 to B0 (f16). At d=0 KV is tiny so this isolates VEC(q8_0) vs TILE(f16) kernel cost.
If q8_0 tg ~= f16 tg, q8_0 is free → ship it (memory headroom for 262144 ctx). NOTE: `-ctk`/`-ctv`
must be equal (build has no FA_ALL_QUANTS); do NOT bench mixed f16/q8_0.

### B4 — decode at depth + KV type (the agentic payoff). Two runs.
```
# q8_0 KV (ship): adds an optional 131072 point that f16 can't reach without OOM
$RUN_BIN/llama-bench -m $M -ngl 99 -fa 1 -ub 4096 -b 8192 -ctk q8_0 -ctv q8_0 -n 128 -d 0,16384,32768,65536 -r 2
# f16 KV baseline for the same depths (cap 65536; 131072 OOMs at f16)
$RUN_BIN/llama-bench -m $M -ngl 99 -fa 1 -ub 4096 -b 8192 -ctk f16 -ctv f16 -n 128 -d 0,16384,32768,65536 -r 2
```
Expect q8_0 to pull ahead as depth grows (half KV bandwidth). This is the real QUEST use case
(deep-research long contexts). Optional q8_0-only deep point: append `-d 131072` to the first run.

### B5 — build A/B: stock SYCL vs MoE-reorder tree (PROVEN; confirm transfer)
```
$BIN/llama-bench    -m $M -ngl 99 -fa 1 -ub 4096 -b 8192 -p 512 -n 128 -r 3   # main / stock
$RUN_BIN/llama-bench -m $M -ngl 99 -fa 1 -ub 4096 -b 8192 -p 512 -n 128 -r 3   # MoE (== B0)
```
Expect MoE tree ~ +16% pp / +6.8% tg over main, matching the baseline ratio. Ship = MoE tree.

### B6 — SPECULATIVE: oneDNN GEMM for prefill (`DISABLE_DNN=0`)
```
GGML_SYCL_DISABLE_DNN=0 $RUN_BIN/llama-bench -m $M -ngl 99 -fa 1 -ub 4096 -b 8192 -p 512,2048 -n 128 -r 3
```
Ship runs DISABLE_DNN=1 (native GEMM, deterministic decode). oneDNN MAY raise pp512/pp2048. If it
beats B0/B1 pp by a real margin AND tg is unharmed, consider DNN=0 for a prefill-bound serving
profile — but ship default stays DNN=1 unless the win is large (decode determinism is the gate reason).

### B7 — SPECULATIVE: topk-moe router fusion A/B (confirm it fires on QUEST 128E/no-bias)
```
GGML_SYCL_DISABLE_TOPK_MOE=1 $RUN_BIN/llama-bench -m $M -ngl 99 -fa 1 -ub 4096 -b 8192 -p 512 -n 128 -r 3
```
Compare tg128 to B0. Expect B0 (fusion on) ~+2-3% tg. If identical, the fusion is NOT engaging on
QUEST's router graph (investigate detection) — but it's default-on so no action needed either way.

### B8 — SPECULATIVE-low: SYCL graph capture (`DISABLE_GRAPH=0`)
```
GGML_SYCL_DISABLE_GRAPH=0 $RUN_BIN/llama-bench -m $M -ngl 99 -fa 1 -ub 4096 -b 8192 -n 128 -d 0,16384 -r 2
```
Decode is exec-bound (per b311f6065), so likely inert; cheap to confirm. Keep only if it helps tg
without hurting pp.

### B9 — SPECULATIVE-low: thread count (expect flat at full offload)
```
$RUN_BIN/llama-bench -m $M -ngl 99 -fa 1 -ub 4096 -b 8192 -t 8,16,32 -p 512 -n 128 -r 2
```

### B10 — SPECULATIVE-low: logical batch control (expect flat for -b>=ub)
```
$RUN_BIN/llama-bench -m $M -ngl 99 -fa 1 -ub 4096 -b 4096,8192 -p 2048 -n 0 -r 3
```

## RECOMMENDED FINAL FLAGS (pre-bench best estimate; B1/B6 may bump `-ub`/DNN)
**Bench/headline (reproduce target):**
```
$MOE_BIN/llama-bench -m $M -ngl 99 -fa 1 -ub 4096 -b 8192 -p 512 -n 128 -r 3
```
**Serve (Phase 6, long agentic ctx) — add q8_0 KV for memory headroom at 262144:**
```
GGML_SYCL_DISABLE_DNN=1 $MOE_BIN/llama-server -m $M -ngl 99 -fa 1 -ub 4096 -b 8192 \
  -ctk q8_0 -ctv q8_0 --jinja   # tools + <think>; port :8093
```
Binary = MoE-reorder tree ($MOE_BIN == $RUN_BIN). Flags: `-ngl 99 -fa 1 -ub 4096 -b 8192`,
KV `q8_0`, `GGML_SYCL_DISABLE_DNN=1`, topk-moe fusion default-ON, `-t 16` (default, irrelevant at
full offload). Revisit `-ub` if B1 shows 2048 wins on 128E; revisit DNN if B6 shows a clean pp win.
```
```
