# QUEST-30B-RL Turbo — audit & campaign plan

**Goal:** a **SYCL/Turbo** B70 packaging of [`osunlp/QUEST-30B-RL`](https://huggingface.co/osunlp/QUEST-30B-RL)
following a proven, repeatable quantization playbook:
**fetch → BF16 GGUF → imatrix → quant sweep → accuracy/agentic gate → speed-tune → serve → release.**

## What QUEST is (verified from config.json + chat_template + card, 2026-06-22)

| Property | Value |
|---|---|
| Org / lineage | osunlp (Ohio State NLP); base = **Qwen3-30B-A3B** (orig. Apr-2025 Qwen3 MoE) |
| Training | mid-training → SFT → **RL** (RL ckpt ships); for **deep research / agentic web interaction** |
| Arch | `qwen3_moe` / `Qwen3MoeForCausalLM` — **MoE** (card says "dense" — *wrong*, config disagrees) |
| Size | ~31B total / ~3.3B active |
| Layers / hidden | 48 / 2048 |
| Attention | 32 heads / 4 KV (GQA), head_dim 128 — **standard Qwen3 attn, NOT gated-delta-net** |
| Experts | **128 total / 8 active**, moe_intermediate 768, **no shared expert** |
| Context | **262144** (2× the prior campaign's 131072) |
| rope_theta | 1e7 · vocab 151936 · BF16 |
| Format | ChatML `<|im_start|>/<|im_end|>`, `<think>` reasoning, **full tool-calling** (`<tool_call>`/`<tool_response>`, `tool` role) |
| Card benches | GAIA 69.0 · LiveResearchBench 74.1 · DeepResearch Bench 45.3 (avg@3) · Apache-2.0 |
| Repo | 13 safetensors shards, ~61.1 GB |

## The point of this campaign (how it differs from the prior campaign)

Same family thesis: **on the B70 speed is not the hard problem** — an A3B MoE already runs ~85–94 t/s.
The deliverable is **the smallest/fastest quant that preserves behavior within a few points of BF16.**

Two things make QUEST's fidelity problem *different* from the prior campaign's:

1. **It's an RL'd *agentic* model, not a math-reasoner.** Its value is multi-step research +
   **tool-calling** (GAIA / LiveResearchBench / DeepResearch). Quant damage on RL models often shows up
   as *behavioral* drift — malformed `<tool_call>` JSON, premature stopping, lost format discipline —
   which **PPL and even a math-probe gate can miss**. So the ship gate must exercise **tool-call
   well-formedness + multi-step research behavior**, not just verifiable arithmetic.
2. **No published quant-sensitivity number.** A prior campaign handed us a 20-pt GPQA@Q4 cliff to defend against.
   QUEST gives us nothing — **we must measure the cliff ourselves** (BF16 vs Q4/Q5/Q6/Q8 on an
   agentic+reasoning gate) before choosing the ship quant. That measurement is the scientific core.

## What transfers from the B70 MoE work (and what doesn't)

- ✅ **Transfers:** `mul_mat_id` Q6_K SYCL reorder; topk-moe **router fusion**; **expert-combine**
  fusion; prefill `-ub 4096` expert-GEMM un-starve; `GGML_SYCL_DISABLE_DNN=1` deterministic decode.
  All are generic to qwen3_moe `mul_mat_id` routing.
- ❌ **Does NOT transfer:** the prior campaign's **gated-delta-net decode fusion** — QUEST uses standard GQA attention,
  so that patch is inapplicable (don't expect the prior campaign's exact +9.8% decode number to reproduce).
- ⚠️ **Re-validate:** imatrix **expert coverage** — 128 experts (vs the prior campaign's 256) is easier, but still needs
  a broad calib corpus; confirm no expert is left uncalibrated before trusting the quants.

## Plan / phases

0. **Baseline** — record card accuracy floor + same-family t/s reference (an A3B MoE already on disk).
1. **Fetch → BF16** — `scripts/01_fetch_source.sh` (~61 GB DL + convert; main converter handles qwen3_moe).
2. **imatrix** — over a wide calib corpus; verify 128-expert coverage.
3. **Quant sweep** — imatrix Q4_K_M / Q5_K_M / Q6_K / Q8_0.
4. **Accuracy + AGENTIC gate** — PPL/parity (necessary-not-sufficient) **plus** a tool-call + research
   behavior gate vs BF16/Q8 reference → **the ship decision** (find the cliff, ship smallest quant above it).
5. **Speed-tune** — llama-bench batch/FA/KV-quant; main vs MoE-reorder build; prefill `-ub` sweep.
6. **Serve** — llama-server on **:8093**, `--jinja` (think + tools) + FA + KV q8_0.
7. **Release** — model card + GGUFs (do **not** push public without explicit user OK).

## Open questions to resolve as we go
- Where's the cliff? (the central unknown — drives the ship quant)
- Does `-ub 4096` prefill win hold at 262144 ctx, or does KV pressure change the optimum?
- Tool-call JSON integrity under Q4 — does it stay well-formed, or is that the first thing to break?
