# QUEST-30B-RL certification suite

A spec-driven, one-command gate that decides whether a quant of **QUEST-30B-RL**
ships. It generalizes `scripts/04b_probes.py` into a unit-testable library
(`lib/harness.py`) + a single source of truth (`spec.yaml`) + a pytest verdict
engine + a GPU orchestrator (`certify.sh`).

The ship decision is **behavioral**: QUEST is an RL'd deep-research / tool-using
model, so quant damage shows up first as wrong tool routing, malformed tool-call
JSON, or lost `<think>` format discipline — which perplexity alone misses. The
gate grades a candidate quant against a hand-verified key **and** against the
BF16 reference's own answers.

## TL;DR — certify a quant in one command

```bash
# Q5_K_M (the ship quant) vs the BF16 agentic-gate reference:
bash tests/certify.sh --quant Q5_K_M

# Re-use pre-computed phase-04/05 logs (skips the GPU ppl/bench runs).
# NOTE: --bench-file must be a CLEAN single-table llama-bench of THIS quant.
# (Do not use results/00_baseline.txt — it's the 35B reference and concatenates two
#  tables; select_bench picks the first/slowest and would falsely NO-SHIP.)
bash tests/certify.sh --quant Q5_K_M \
  --ppl-file results/04_ppl_Q5.txt --bench-file results/05_build_moe.txt
```

Exit code is the verdict: **0 = SHIP**, 1 = a gate failed, 2 = bad args / missing
model, 3 = server health timeout (GPU busy / OOM). `certify.sh` is the **only**
component that touches the B70; it owns exactly one `llama-server`, runs any
needed `llama-perplexity` / `llama-bench` as serialized short-lived processes
(one GPU job at a time), and always tears the server down via a SIGTERM ->
grace -> SIGKILL -> reap trap, so it never leaks VRAM.

## The interpreter

Any Python 3.10+ with the test dependencies works. `env.sh` uses `$HOME/venv/bin/python`
when that file exists, and otherwise `python3` on `PATH`.

```bash
python3 -m pip install -r tests/requirements.txt
python3 -m pytest -q
```

GPU certification (`tests/certify.sh`) uses the same `CONV_PY`.

## What runs with NO GPU

The suite is **gated**: behavioral / metadata / perf / ppl tests `skip()` unless
their CLI option is supplied, so you can introspect everything without a GPU.

| command | what happens |
| --- | --- |
| `python3 -m pytest -q` | collects the whole suite, **runs only the pure unit tests** (synthetic, in-memory, plus the committed logs). Gated GPU tests SKIP. |
| `python3 -m pytest --collect-only` | lists the full suite (behavioral + metadata + perf + ppl) without touching the GPU. |
| `python3 -m pytest tests/test_scoring.py -q` | the pure scorer/certify unit tests only. No server, no model. |

Gates (each test that needs it SKIPs when the option is absent):

| option | unlocks | reads |
| --- | --- | --- |
| `--server-url URL` | behavioral probes (`test_agentic`) | live `llama-server` (OpenAI `/v1/chat/completions`) |
| `--model-path PATH` | GGUF topology (`test_metadata`) | GGUF **header only** (mmap KV store; does NOT load tensors or touch the GPU) |
| `--ref-answers PATH` | agreement + `certify()` | BF16 reference JSON (`results/04b_answers/BF16.json`) |
| `--bench-file PATH` | performance gate (`test_performance`) | a saved `llama-bench` markdown table |
| `--ppl-file PATH` | quality gate (`test_perplexity`) | a saved `llama-perplexity` log |
| `--spec PATH` | override the spec | `tests/spec.yaml` by default |
| `--perf-tolerance F` | perf min tolerance | `None` -> `spec.thresholds.perf.tolerance` (0.05) |
| `--out-dir DIR` | where artifacts (e.g. `certify_<quant>.json`) are written | — |

`--bench-file` / `--ppl-file` fixtures return the **file contents (str)**, fed
straight into `harness.parse_llama_bench` / `harness.parse_ppl`.

## What PASS means (the ship rule)

`harness.certify(cand, ref, thresholds)` is the single source of the ship/no-ship
boolean (asserted by `test_agentic::test_certify`). Order:

1. **SANITY GUARD (on the BF16 reference).** If BF16 `tool_key < 8/9` or
   `think_ok < 10/10`, the **rig** is broken (template / `--reasoning-format` /
   tool schema), not the model — `passed=False`, fix the rig first.
2. **AGGREGATE tolerances** (all must hold):
   - `tool_key(cand)   >= tool_key(ref) - 1`
   - `tool_agree(cand) >= 8`  (nt-1; disagree with BF16 on <=1 tool probe)
   - `reason_key(cand) >= reason_key(ref) - 2`
   - `reason_agree(cand) >= 8`  (nr-2)
   - `think_ok(ref) == 10` **and** `think_ok(cand) == think_ok(ref)`
3. **HARD FAILS (any one blocks ship):**
   - **H1** — new malformed/unparseable tool-call JSON on a probe BF16 got clean.
   - **H2** — regression on a critical probe `{nested_enum, abstain, par_weather}` BF16 passed.
   - **H3** — any `<think>` structural regression (cand `think_ok` < ref on any probe).

`passed = sanity_ok AND all aggregate checks pass AND no hard fail fired.`

Plus two quality guards: **PPL** Q5 wikitext `<= 8.5` (gross-breakage guard,
anchor 7.19) and **PERF** `pp512 >= 1096.9` / `tg128 >= 75.9` (within 5% of the
B70 MoE-reorder ceiling 1154.6 / 79.9; `round(1154.6*0.95,1)=1096.9`,
`round(79.9*0.95,1)=75.9`).

**abstain note:** it is failed by BOTH BF16 and Q5 (base-model tool-eagerness).
`agreement()` counts mutual pass only, so abstain contributes 0 to `tool_agree`
either way (max attainable is 8/9 == agree_min), and H2 is guarded on ref-pass,
so abstain can never fire a regression. No special-casing needed.

## Layout

```
tests/
  lib/__init__.py        package marker
  lib/harness.py         the ONE reusable module (pure scorers + server lifecycle + parsers + certify)
  conftest.py            CLI options, gated fixtures, marker registration, sys.path wiring
  spec.yaml              SINGLE SOURCE OF TRUTH: 9 tool + 10 reason probes, schemas, metadata, thresholds
  requirements.txt       pinned deps (install into $CONV_PY)
  certify.sh             one-command GPU orchestrator (the only thing that touches the B70)
  README.md              this file
  test_scoring.py        PURE unit tests (synthetic data; the only thing bare pytest fully RUNS)
  test_agentic.py        behavioral cert (gated on --server-url)
  test_metadata.py       GGUF topology (gated on --model-path)
  test_performance.py    perf gate (gated on --bench-file)
  test_perplexity.py     quality gate (gated on --ppl-file)
```

`spec.yaml` is data-only. The 04b lambdas became a declarative arg-check DSL
(`equals` / `contains_all` / `contains_any` / `type` / `min_len` /
`items_have_keys`, with dot-nested `field` paths), so `check_tool_args` /
`score_tool` are pure and unit-testable. Everything else
(`extract_tool_call`, `extract_answer`, `normalize_answer`, `think_ok`, the
abstain/parallel/`min_calls` pass logic, and the mutual-pass agreement counting)
follows `scripts/04b_probes.py`. Parallel probes (`min_calls >= 2`) now require
every call to be well-formed, not only `tool_calls[0]`. Inline `<tool_call>`
JSON is parsed with `JSONDecoder.raw_decode`, so nested arguments are not cut
at the first `}`. Existing `results/04b_answers/*.json` still drop straight in
as `--ref-answers` (those files already store scored records).

## Regenerating the BF16 reference answers

The reference is the agentic-gate's BF16 output, `results/04b_answers/BF16.json`,
in the exact `{tool:{id:rec}, reason:{id:rec}}` shape `harness.collect()` emits.
To regenerate it against a freshly served BF16 (needs the GPU; BF16 oversizes
VRAM so it uses partial offload via `ngl_for`):

```bash
source env.sh
# serve BF16, then:
python - <<'PY'
import sys; sys.path.insert(0, "tests/lib")
import harness, json
spec = harness.load_spec()
ans = harness.collect("http://127.0.0.1:8099", spec, max_tokens=spec["meta"]["max_tokens"])
json.dump(ans, open("results/04b_answers/BF16.json", "w"), indent=2)
PY
```

(or just run the existing `scripts/04b_agentic_gate.sh` with `MODELS=BF16` — its
output shape is byte-compatible.)
