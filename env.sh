#!/usr/bin/env bash
# Shared environment for the QUEST-30B-RL Turbo campaign. Source this, don't execute.
# Verified facts as of 2026-06-22 (from osunlp/QUEST-30B-RL config.json + chat_template + card).
#
# QUEST-30B-RL (osunlp / Ohio State NLP) is a deep-research / agentic web-interaction model:
#   3-stage train (mid-training -> SFT -> RL); RL checkpoint is the ship version. Apache-2.0.
#   Base = Qwen3-30B-A3B (the ORIGINAL Apr-2025 Qwen3 MoE, NOT Qwen3.6-35B).
#   arch = qwen3_moe (Qwen3MoeForCausalLM), MoE, ~31B total / ~3.3B active:
#     48 layers, hidden 2048, 32 attn / 4 KV heads (GQA), head_dim 128,
#     128 experts / 8 active (moe_intermediate 768, dense intermediate 6144, NO shared expert),
#     vocab 151936, ctx 262144, rope_theta 1e7, BF16.
#   Format: Qwen ChatML (<|im_start|>/<|im_end|>), <think> reasoning, FULL tool-calling
#     (<tool_call>/<tool_response>, `tool` role). Card benches: GAIA 69.0, LiveResearchBench 74.1,
#     DeepResearch Bench 45.3.
#
# => Same A3B-MoE *family* as the B70 we already tune, so the SYCL MoE work (mul_mat_id reorder,
#    topk-moe router + expert-combine fusions, prefill -ub) TRANSFERS. Two deltas vs the prior campaign base:
#      - 128 experts (not 256) + no shared expert -> simpler imatrix coverage, different routing.
#      - STANDARD Qwen3 attention (GQA), NOT gated-delta-net -> the prior campaign's gated-delta-net decode
#        fusion does NOT apply here; the router/expert-combine fusions still do.

# Intel oneAPI / SYCL runtime (llama binaries link libsvml.so — missing it = "cannot open shared object").
# setvars.sh references unbound vars and trips `set -u`/`set -e`, so save shell opts, relax, restore.
__q_oldopts="$(set +o)"
set +eu
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
eval "$__q_oldopts"; unset __q_oldopts
# setvars sets DNNLROOT but doesn't add oneDNN headers to CPATH; rebuilding ggml-sycl needs them.
[ -n "${DNNLROOT:-}" ] && [ -d "$DNNLROOT/include" ] && export CPATH="$DNNLROOT/include:${CPATH:-}"

# --- llama.cpp trees -------------------------------------------------------
# Main SYCL build (general purpose). Its converter already registers Qwen3MoeForCausalLM
# (conversion/qwen.py) -> QUEST converts with the MAIN tree; no MoE-tree fallback needed.
export LCPP="${LCPP:-$HOME/llama.cpp}"
export BIN="$LCPP/build/bin"
# MoE-tuned tree: Q6_K mul_mat_id SYCL reorder + the decode/prefill fusion patches (B70 speed).
export MOE_LCPP="${MOE_LCPP:-$HOME/llama.cpp-sycl-moe-ready}"
export MOE_BIN="$MOE_LCPP/build/bin"
# Prefer the MoE tree's binaries if built, else the main build.
if [[ -x "$MOE_BIN/llama-server" ]]; then
  RUN_BIN="$MOE_BIN"
else
  RUN_BIN="$BIN"
fi
export RUN_BIN

# --- paths -----------------------------------------------------------------
# The campaign directory is this repo. Override WORK when artifacts live elsewhere.
_QUEST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_DIR="${MODEL_DIR:-${HOME:-}/models/QUEST-30B-RL}"
export WORK="${WORK:-$_QUEST_ROOT}"
export ART="$WORK/artifacts"          # bf16 + imatrix + quant outputs (big, gitignored)
export RES="$WORK/results"            # bench / ppl / parity / gate logs (committed)

export BF16="$ART/quest-30b-rl-bf16.gguf"   # high-precision source (model ships BF16)
export IMATRIX="$ART/quest-30b-rl.imatrix"
export CALIB="${CALIB:-$HOME/calib-data/calibration_datav3.txt}"  # reused shared corpus
# MoE imatrix coverage: 128 experts x 8-active need a broad corpus or some experts go uncalibrated.
# Validate expert coverage before trusting a low-chunk run (see scripts/02_imatrix.sh).

# Same-FAMILY speed reference (a different A3B MoE already on disk) — sanity-check t/s ONLY,
# never an accuracy ref (different weights AND topology: 35B/256-expert vs QUEST 30B/128-expert).
export BASE_REF="${BASE_REF:-$HOME/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf}"

# --- HF / Python ----------------------------------------------------------
# Prefer an existing venv, then whatever is on PATH. A missing $HOME/venv used to
# be the only default, so a fresh clone could not run the cert suite.
if [[ -z "${HF_BIN:-}" ]]; then
  if [[ -n "${HOME:-}" && -x "$HOME/venv/bin/hf" ]]; then
    HF_BIN="$HOME/venv/bin/hf"
  elif command -v hf >/dev/null 2>&1; then
    HF_BIN="$(command -v hf)"
  else
    HF_BIN="hf"
  fi
fi
export HF_BIN
if [[ -z "${CONV_PY:-}" ]]; then
  if [[ -n "${HOME:-}" && -x "$HOME/venv/bin/python" ]]; then
    CONV_PY="$HOME/venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    CONV_PY="$(command -v python3)"
  else
    CONV_PY="python3"
  fi
fi
export CONV_PY
export HF_REPO="osunlp/QUEST-30B-RL"
export HF_RELEASE="Frosty40/QUEST-30B-RL-B70-Turbo-GGUF"     # (release target; do NOT push w/o user OK)

# --- runtime --------------------------------------------------------------
export NGL="${NGL:-99}"        # 48 layers -> 99 offloads everything to the B70
export PORT="${PORT:-8093}"    # QUEST serves on :8093
# Reasoning + agentic model: spends tokens inside <think>. Smoke/parity gens need a real budget
# or replies come back empty. Greedy (temp 0) for verifiable gates; card default temp 0.7 for agentic.
export GEN_N="${GEN_N:-512}"

# Ship config (B70 Turbo). DISABLE_DNN -> deterministic decode + native GEMM; UBATCH/BBATCH = prefill win.
export GGML_SYCL_DISABLE_DNN="${GGML_SYCL_DISABLE_DNN:-1}"
export UBATCH="${UBATCH:-4096}"   # -ub: un-starves MoE expert GEMMs (prefill win, validated on the prior campaign)
export BBATCH="${BBATCH:-8192}"   # -b: logical batch, must be >= UBATCH

# VRAM-aware offload: the B70 has ~30 GiB. A3B quants fit fully (Q4 ~17 / Q5 ~21 / Q6 ~25 GB -> -ngl 99);
# Q8_0 (~32) and BF16 (~61) oversize -> partial -ngl so imatrix/ppl don't OOM. 48 transformer layers.
export VRAM_GIB="${VRAM_GIB:-30}"
export N_LAYERS="${N_LAYERS:-48}"
ngl_for() {
  local f="$1" sz_mib per_mib budget_mib n
  sz_mib=$(( $(stat -c%s "$f" 2>/dev/null || echo 0) / 1048576 ))
  [ "$sz_mib" -le 0 ] && { echo 99; return; }                 # unknown -> try full
  if [ "$sz_mib" -le $(( (VRAM_GIB - 3) * 1024 )) ]; then echo 99; return; fi
  budget_mib=$(( (VRAM_GIB - 6) * 1024 ))                      # reserve ~6 GiB for KV/ctx/compute
  per_mib=$(( sz_mib / N_LAYERS )); [ "$per_mib" -le 0 ] && per_mib=1
  n=$(( budget_mib / per_mib )); [ "$n" -lt 1 ] && n=1; [ "$n" -gt 99 ] && n=99
  echo "$n"
}

# llama.cpp renamed the completion example to llama-cli. Accept either binary.
llama_completion_bin() {
  local d
  for d in "$RUN_BIN" "$BIN" "$MOE_BIN"; do
    [[ -z "$d" ]] && continue
    if [[ -x "$d/llama-cli" ]]; then echo "$d/llama-cli"; return 0; fi
    if [[ -x "$d/llama-completion" ]]; then echo "$d/llama-completion"; return 0; fi
  done
  return 1
}

mkdir -p "$ART" "$RES"
