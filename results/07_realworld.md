# Phase 7 — real-world end-to-end test (Q5_K_M, served, 2026-06-22)

Measured by serving `quest-30b-rl-Q5_K_M.gguf` (ship config: `-ngl 99 -fa on --cache-type-k/v q8_0
-ub 4096 -b 8192 --jinja`, ctx up to 65536) and making real `/v1/chat/completions` calls.
These supersede the synthetic `llama-bench` figures for any user-facing speed claim.

## Throughput (authoritative server-side timings + end-to-end wall-clock)

| Scenario | prompt tok | gen tok | prefill (tok/s) | decode (tok/s) | e2e (tok/s) |
|---|---:|---:|---:|---:|---:|
| short context | 30 | 320 | (n/a, tiny) | **85.8** | 74.0 |
| ~8K context | 7,774 | 64 | **1,377** | **30.8** | 8.3* |
| 5-turn chat (decode, stable) | — | 149–500 | — | **81–85** | 65–81 |

*the 8K "e2e 8.3" is dominated by the 5.6 s prefill of 7,774 tokens; pure decode at that depth is 30.8 tok/s.

- **Prefill ~1,380 tok/s is REAL** (1,377 measured over 7,774 actual tokens — matches llama-bench).
- **Decode ~80–86 tok/s at short context** (stable across a 5-turn session), **~31 tok/s at 8K**, lower deeper.
- The earlier "up to 109 tok/s" was the KV-**f16** `llama-bench` config at **empty context** — NOT the ship
  config and NOT representative. Corrected.
- Fast *for a 30B* only because it's a 3.3B-active MoE; long-context decode is attention-bound.

## Behavior (5-turn real conversation)

- **Reasoning**: correct (MoE active-param math: 8/128 × 30B ≈ 1.875B), coherent multi-sentence answers.
- **Tool routing**: ✅ chose `calculator` with `{"expression":"450 / 2.3"}`; consumed the `195.65`
  result correctly on the round-trip; no malformed JSON.
- **Multi-turn memory**: ✅ referenced the prior tool result and an earlier figure across turns.
- **Operational caveat (confirmed)**: heavy `<think>` usage — at `max_tokens=500`, two reasoning turns
  consumed the entire budget inside `<think>` and emitted NO final answer. At `max_tokens=2048` it
  produced a complete `<think>` + answer (352 tok, 74 tok/s). **Budget ≥1,500 for reasoning tasks.**

Raw logs: `results/rt_probe_run.log`, `results/rt_chat_run.log`, `results/rt_full_run.log`.
