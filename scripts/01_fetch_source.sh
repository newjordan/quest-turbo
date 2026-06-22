#!/usr/bin/env bash
# Phase 1 — acquire the high-precision source: QUEST-30B-RL safetensors -> BF16 GGUF.
# ~61 GB download (13 shards) + ~61 GB BF16 output. IO/CPU heavy; run when the fleet is idle.
set -euo pipefail
source "$(dirname "$0")/../env.sh"

SRC_DIR="$ART/hf-src"
mkdir -p "$SRC_DIR" "$MODEL_DIR"

echo ">> Downloading QUEST checkpoint: $HF_REPO (everything except any pre-baked GGUFs)"
# hf 1.18 GOTCHA: --include takes ONE pattern and treats extras as positional filenames (silently
# drops the safetensors). Use a single --exclude. QUEST has no stock GGUFs, but keep the guard.
"$HF_BIN" download "$HF_REPO" --local-dir "$SRC_DIR" --exclude "*.gguf"

echo ">> Converting to BF16 GGUF -> $BF16  (via $CONV_PY)"
"$CONV_PY" -c 'import torch' 2>/dev/null \
  || { echo "ERROR: $CONV_PY lacks torch — set CONV_PY to a venv that has it"; exit 1; }
# Main converter registers Qwen3MoeForCausalLM (conversion/qwen.py) -> should convert directly.
# Keep the MoE-tree converter as a fallback; fail HARD if both fail.
"$CONV_PY" "$LCPP/convert_hf_to_gguf.py" "$SRC_DIR" --outfile "$BF16" --outtype bf16 \
  || { echo "   main converter failed — retrying with MoE tree converter"; \
       "$CONV_PY" "$MOE_LCPP/convert_hf_to_gguf.py" "$SRC_DIR" --outfile "$BF16" --outtype bf16; }
[[ -s "$BF16" ]] || { echo "ERROR: conversion produced no $BF16"; exit 1; }

echo ">> Verifying BF16 GGUF (CPU-only; BF16 is ~61GB and won't fit the B70 VRAM for a GPU smoke)"
ls -lh "$BF16" | tee "$RES/01_bf16_info.txt"
"$CONV_PY" - "$BF16" <<'PY' 2>/dev/null | tee -a "$RES/01_bf16_info.txt" || true
import sys, gguf
r = gguf.GGUFReader(sys.argv[1])
def val(k):
    f = r.fields.get(k)
    if not f: return "?"
    try: return bytes(f.parts[f.data[0]]).decode()
    except Exception: return f.parts[f.data[0]]
print("architecture   =", val("general.architecture"))
print("name           =", val("general.name"))
print("expert_count   =", val("qwen3moe.expert_count"))
print("block_count    =", val("qwen3moe.block_count"))
print("context_length =", val("qwen3moe.context_length"))
print("tensors        =", len(r.tensors))
PY

echo "Phase 1 done. BF16 at: $BF16"
echo "TIP: after Phases 2-4 verify, delete $SRC_DIR to reclaim ~61 GB."
