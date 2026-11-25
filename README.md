# QUEST-30B-RL Turbo

![QUEST-30B-RL · B70 Turbo](release/hero.jpg)

Intel Arc (B70) / `llama.cpp` SYCL packaging of
[`osunlp/QUEST-30B-RL`](https://huggingface.co/osunlp/QUEST-30B-RL): fetch, importance-matrix,
quantize, an agentic ship gate, speed notes, and a serve script. The quant this tree ships is
**Q5_K_M**. Model weights are not in git (`*.gguf` is ignored). Nothing in the test suite
downloads them.

The ship decision is behavioral. QUEST is an RL deep-research model, so a bad quant shows up
as a wrong tool, broken tool-call JSON, or a lost `<think>` / `ANSWER:` shape. Perplexity is
only a gross-breakage check.

## Setup

Python 3.10 or newer. From the repo root:

```bash
python3 -m pip install -r tests/requirements.txt
python3 -m pytest -q
```

A run of that pytest command in this tree reported **115 passed, 35 skipped**. The skips are the
server, GGUF-header, and saved-log gates; they run when `certify.sh` passes the matching flags.

`source env.sh` before the phase scripts. It sets paths, the llama.cpp binary directory, and
`ngl_for` (how many layers fit in `VRAM_GIB`, default 30). `WORK` defaults to this repo.
Override it when the GGUFs live somewhere else:

```bash
export WORK="$HOME/quest-turbo"   # artifacts/ and results/ live here instead
source env.sh
```

`CONV_PY` is `$HOME/venv/bin/python` when that file exists, otherwise `python3` on `PATH`.
`HF_BIN` is chosen the same way.

GPU phases need a SYCL `llama.cpp` build (`llama-server`, `llama-cli`, `llama-bench`,
`llama-perplexity`, `llama-quantize`, `llama-imatrix`). `RUN_BIN` prefers
`$MOE_LCPP/build/bin` (`$HOME/llama.cpp-sycl-moe-ready`) and falls back to `$LCPP/build/bin`
(`$HOME/llama.cpp`). Intel oneAPI is sourced when `/opt/intel/oneapi/setvars.sh` exists.
Phase 1 also needs `torch` and `gguf` in `CONV_PY`, and the `hf` CLI.

## Offline commands

These do not start a server and do not read a GGUF unless you pass one.

```bash
python3 -m pytest -q

python3 tests/lib/harness.py certify \
  --cand results/04b_answers/Q5_K_M.json \
  --ref  results/04b_answers/BF16.json

python3 scripts/04b_probes.py grade results/04b_answers BF16 Q5_K_M

python3 tests/lib/harness.py ppl   results/04_ppl_Q5.txt
python3 tests/lib/harness.py bench results/05_depth.txt
```

`certify` / `grade` exit 0 for SHIP and 1 for NO-SHIP. `NO_COLOR=1` turns off ANSI color.
Probe text, tool schemas, and thresholds are `tests/spec.yaml`.

## GPU pipeline

| Script | What it does |
|---|---|
| `scripts/00_baseline_bench.sh` | Bench a same-family GGUF already on disk (`BASE_REF`). Not QUEST. |
| `scripts/01_fetch_source.sh` | Download `osunlp/QUEST-30B-RL` and convert to BF16. Large download. |
| `scripts/02_imatrix.sh` | Importance matrix from `CALIB`. |
| `scripts/03_quant_sweep.sh` | Quantize. `QUANTS` defaults to `Q4_K_M Q5_K_M Q6_K Q8_0`. |
| `scripts/04_accuracy.sh` | Perplexity, then greedy parity. A failed or empty generation is `ERROR`, not `EXACT`. |
| `scripts/04b_agentic_gate.sh` | Serve each quant, collect probes, grade against BF16. Reaps the server on failure. |
| `scripts/05_focused.sh` | Q5 speed notes (batch, flash attention, depth). |
| `scripts/05_speed_tune.sh` | Wider sweep on the ship binary (`RUN_BIN`). |
| `scripts/06_serve.sh` | `llama-server` on `127.0.0.1:$PORT` (default port 8093). |
| `scripts/06_smoke.sh` | Tool route, tool round-trip, and a reasoning reply. Non-zero if a check fails. |
| `tests/certify.sh` | One GPU owner: optional perplexity, optional bench, then pytest. |

```bash
source env.sh
bash scripts/04b_agentic_gate.sh          # MODELS="Q5_K_M BF16" to limit the set

# Full GPU cert. Exit 0 = SHIP, 1 = a gate failed, 2 = bad args / missing model, 3 = health timeout.
bash tests/certify.sh --quant Q5_K_M \
  --ppl-file results/04_ppl_Q5.txt \
  --bench-file results/05_depth.txt

HOST=127.0.0.1 QUANT=Q5_K_M CTX=65536 KV=q8_0 bash scripts/06_serve.sh
```

`06_serve.sh` binds localhost. Set `HOST=0.0.0.0` only on a trusted network and put extra
server flags in `SERVE_EXTRA` (for example an `--api-key`). `--jinja` is required for
`<think>` and tool calling. Reasoning replies need a large `max_tokens` (the spec uses 4096);
the served logs below show a turn that spent the whole budget inside the trace when the cap
was small.

`llama-bench` tables with two depth-0 rows for the same test (flash attention on and off, or
two builds concatenated) are scored from the **first** row. `results/00_baseline.txt` and
`results/05_fa.txt` are those files. `harness.py bench` prints a warning. Pass
`results/05_depth.txt` for the KV-q8_0 ship table.

## Ship rule

`harness.certify` is the decision (`tests/spec.yaml`):

1. The BF16 reference itself must score at least 8/9 tools and 10/10 think checks. Below that, the rig is broken.
2. Tool key may drop by at most 1, and at least 8 tool probes must agree with BF16. Reason key may drop by at most 2, with at least 8 agreements. Think count must stay equal to the reference.
3. Hard fails: new unparseable tool JSON (H1), a regression on `nested_enum` / `abstain` / `par_weather` when BF16 passed (H2), or a think-structure regression (H3).

`abstain` fails for both BF16 and Q5 in the saved answers (the model calls a tool anyway). That shared miss is not a regression. Parallel probes require every call to be well-formed.

Perplexity ceiling is 8.5. The speed floor is 5% under the **35B reference** bench in `results/00_baseline.txt` (1154.6 and 79.9 tok/s), which is 1096.9 and 75.9. It is a floor, not QUEST's own speed.

## Recorded logs

These figures are parsed from the files in `results/`. They were not re-measured on a GPU for this tree.

Agentic gate (`results/04b_summary.txt`, `results/04b_answers/`): BF16 and Q5_K_M both score TOOL 8/9 (agree 8/9), REASON 10/10 (agree 10/10), THINK 10/10. `harness.certify` on those two JSON files is SHIP.

Perplexity (`results/04_ppl_Q5.txt`): wikitext-2, 40 chunks, final estimate **7.1905 +/- 0.20116**. The BF16 imatrix calib line in `results/02_coverage_summary.txt` is **7.4375 +/- 0.10953**. That file also records no dead experts and 42 of 144 expert tensors with partial coverage.

File sizes (`results/03_quant_sizes.txt`; the "GB" column there is 1024^3):

| Quant | Bytes | GiB |
|---|---:|---:|
| Q4_K_M | 18,556,686,048 | 17.28 |
| Q5_K_M | 21,725,581,024 | 20.23 |
| Q6_K | 25,092,531,936 | 23.37 |

20.23 GiB is 21.73 GB (decimal). Q5_K_M is the file in the Hugging Face repo below.

`results/05_depth.txt` is Q5_K_M, `-ngl 99`, FA on, `-ub 4096`, `-b 8192`, KV q8_0:

| Test | tok/s |
|---|---:|
| pp512 | 1379.54 ± 9.50 |
| tg128 | 89.94 ± 0.33 |
| pp512 @ 16384 | 418.72 ± 1.31 |
| tg128 @ 16384 | 17.73 ± 0.02 |
| pp512 @ 32768 | 240.04 ± 0.47 |
| tg128 @ 32768 | 9.92 ± 0.01 |

`results/05_ub.txt` prefill at 4096 tokens, same quant: ubatch 512 → 1130.19 ± 6.16, ubatch 2048 → 1702.95 ± 1.28, ubatch 4096 → 1803.74 ± 2.24.

Served timings in `results/rt_server.log` (same numbers copied in `results/rt_probe_run.log`):

- 320 generated tokens in 3728.68 ms → **85.82 tok/s** (30 prompt tokens).
- 7774 prompt tokens in 5645.57 ms → **1377.01 tok/s** prefill.
- 64 generated tokens after that prompt in 2076.76 ms → **30.82 tok/s**.

`results/05_build_moe.txt` is a different llama-bench row (pp512 1392.88 ± 14.85, tg128 109.65 ± 0.18) with no KV q8_0 columns. Use the depth file and the server log above for the q8_0 configuration.

The 35B reference in `results/00_baseline.txt` (not this model): MoE-reorder build pp512 1154.63 ± 4.09, tg128 79.91 ± 0.07. Main build pp512 993.37 ± 2.70, tg128 74.83 ± 0.03.

## Layout

```
env.sh                 shared paths and ngl_for
scripts/               phase 00–06
tests/spec.yaml        probes and thresholds
tests/lib/harness.py   scorers, parsers, certify, offline CLI
tests/certify.sh       GPU orchestrator
results/               campaign logs (no weights)
release/README.md      Hugging Face model card
```

Published weights: [Frosty40/QUEST-30B-RL-B70-Turbo-GGUF](https://huggingface.co/Frosty40/QUEST-30B-RL-B70-Turbo-GGUF).

`AUDIT.md` is the campaign plan. `LOOP-STATE.md` is the session ledger from the original runs.

## License and credits

Scripts and docs in this repo are **MIT** (`LICENSE`). The model weights are **Apache-2.0**, inherited from QUEST and Qwen. They are not vendored here.

- **Model:** [OSU NLP (`osunlp`)](https://huggingface.co/osunlp), [`QUEST-30B-RL`](https://huggingface.co/osunlp/QUEST-30B-RL). The capability is theirs.
- **Architecture:** [Qwen](https://huggingface.co/Qwen), Qwen3-30B-A3B.
- **Runtime and GGUF:** [`llama.cpp` / ggml-org](https://github.com/ggml-org/llama.cpp).
- **GPU backend:** [Intel oneAPI / SYCL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html).

The layer in this repo (imatrix, Q5_K_M, the agentic gate, the Arc notes, packaging) is by [Frosty40](https://huggingface.co/Frosty40). Cite QUEST and Qwen first.
