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

echo ">> Verifying BF16 GGUF (header only; does not load tensors or the GPU)"
# stat, not ls -l: ls records the local username and home path in the log.
stat -c 'size_bytes=%s' "$BF16" | tee "$RES/01_bf16_info.txt"
"$CONV_PY" - "$BF16" <<'PY' | tee -a "$RES/01_bf16_info.txt"
import sys
import gguf

r = gguf.GGUFReader(sys.argv[1])

def show(v):
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    item = getattr(v, "item", None)
    if callable(item) and not isinstance(v, (str, bytes, list, tuple, dict)):
        try:
            return item()
        except Exception:
            pass
    if isinstance(v, list) and len(v) == 1:
        return show(v[0])
    return v

keys = [
    "general.architecture",
    "general.name",
    "qwen3moe.expert_count",
    "qwen3moe.expert_used_count",
    "qwen3moe.block_count",
    "qwen3moe.context_length",
    "qwen3moe.attention.head_count",
    "qwen3moe.attention.head_count_kv",
    "qwen3moe.embedding_length",
    "qwen3moe.feed_forward_length",
    "qwen3moe.expert_feed_forward_length",
    "qwen3moe.rope.freq_base",
]
for key in keys:
    field = r.fields.get(key)
    print(f"{key} = {show(field.contents()) if field else '?'}")
print("tensors =", len(r.tensors))
PY

echo "Phase 1 done. BF16 at: $BF16"
echo "TIP: after Phases 2-4 verify, delete $SRC_DIR to reclaim ~61 GB."
