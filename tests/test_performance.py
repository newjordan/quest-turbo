"""QUALITY GATE — throughput (B70 MoE-reorder speed floor).

AUTHORITATIVE DESIGN (frozen contract): this test file NEVER touches the B70 GPU and
NEVER starts llama-bench. The GPU run happens exactly once, in tests/certify.sh, which
executes

    $RUN_BIN/llama-bench -m <quant>.gguf -ngl 99 -fa 1 -ub 4096 -b 8192 -p 512 -n 128 -r 3
        | tee $RES/certify_bench_<quant>.txt

and passes that saved markdown table to pytest as --bench-file. Here we ONLY parse that
table via harness.parse_llama_bench and assert pp512 / tg128 stay within the configurable
tolerance (--perf-tolerance, default 0.05) of the spec baselines (1154.6 / 79.9 t/s) via
harness.check_performance. With tol=0.05 that floor is pp512 >= ~1097, tg128 >= ~75.9.

Two tiers live in this file:
  * PURE unit tests — feed synthetic llama-bench tables (both the plain
    model|...|ngl|fa|test|t/s layout from results/00_baseline.txt and the wider
    ...|n_batch|n_ubatch|type_k|type_v|fa|test|t/s depth layout from results/05_depth.txt)
    to the import-pure parser / selector / gate functions. These RUN under bare `pytest`
    with no server, no model, no GPU.
  * The gated quality test — skips unless --bench-file is given; then it is a pure parse +
    threshold check of the saved table.
"""
import os

import pytest

import harness


# ===========================================================================
# Synthetic llama-bench tables (real column layouts, real '±' separator).
# ===========================================================================

# Plain layout: model | size | params | backend | ngl | fa | test | t/s
# (mirrors results/00_baseline.txt, MoE-reorder build headline numbers).
_BENCH_PLAIN = """\
| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| qwen3moe 30B.A3B Q5_K - Medium |  20.23 GiB |    30.53 B | SYCL       |  99 |   1 |           pp512 |       1154.63 ± 4.09 |
| qwen3moe 30B.A3B Q5_K - Medium |  20.23 GiB |    30.53 B | SYCL       |  99 |   1 |           tg128 |         79.91 ± 0.07 |

build: 0b98ca1b4 (48)
"""

# Wide layout WITH depth rows: ngl | n_batch | n_ubatch | type_k | type_v | fa | test | t/s
# (mirrors results/05_depth.txt). Headline (no-depth) rows must be selected by default.
_BENCH_DEPTH = """\
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch | type_k | type_v |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -----: | -----: | --: | --------------: | -------------------: |
| qwen3moe 30B.A3B Q5_K - Medium |  20.23 GiB |    30.53 B | SYCL       |  99 |    8192 |     4096 |   q8_0 |   q8_0 |   1 |           pp512 |       1379.54 ± 9.50 |
| qwen3moe 30B.A3B Q5_K - Medium |  20.23 GiB |    30.53 B | SYCL       |  99 |    8192 |     4096 |   q8_0 |   q8_0 |   1 |           tg128 |         89.94 ± 0.33 |
| qwen3moe 30B.A3B Q5_K - Medium |  20.23 GiB |    30.53 B | SYCL       |  99 |    8192 |     4096 |   q8_0 |   q8_0 |   1 |  pp512 @ d16384 |        418.72 ± 1.31 |
| qwen3moe 30B.A3B Q5_K - Medium |  20.23 GiB |    30.53 B | SYCL       |  99 |    8192 |     4096 |   q8_0 |   q8_0 |   1 |  tg128 @ d16384 |         17.73 ± 0.02 |

build: 0b98ca1b4 (48)
"""

# A regression table whose headline numbers are below the spec floor.
_BENCH_SLOW = """\
| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| qwen3moe 30B.A3B Q5_K - Medium |  20.23 GiB |    30.53 B | SYCL       |  99 |   1 |           pp512 |       1000.00 ± 5.00 |
| qwen3moe 30B.A3B Q5_K - Medium |  20.23 GiB |    30.53 B | SYCL       |  99 |   1 |           tg128 |         70.00 ± 0.10 |
"""

# A table missing the pp512 row entirely (only tg128 present).
_BENCH_NO_PP = """\
| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| qwen3moe 30B.A3B Q5_K - Medium |  20.23 GiB |    30.53 B | SYCL       |  99 |   1 |           tg128 |         79.91 ± 0.07 |
"""

# Hand-built thresholds matching resolve_thresholds() output: flat pp512_min/tg128_min.
# (1154.6 * 0.95 ~= 1097, 79.9 * 0.95 ~= 75.9.)
_TH = {"pp512_min": 1097.0, "tg128_min": 75.9}


# ===========================================================================
# PURE unit tests — parse_llama_bench. No GPU / server / model.
# ===========================================================================

def test_parse_plain_table_row_count():
    rows = harness.parse_llama_bench(_BENCH_PLAIN)
    # Two data rows; header / separator / 'build:' lines dropped.
    assert len(rows) == 2


def test_parse_plain_table_fields():
    rows = harness.parse_llama_bench(_BENCH_PLAIN)
    by_test = {r["test"]: r for r in rows}
    assert set(by_test) == {"pp512", "tg128"}
    assert by_test["pp512"]["t_s"] == pytest.approx(1154.63)
    assert by_test["tg128"]["t_s"] == pytest.approx(79.91)
    # ngl / fa coerced to int per contract.
    assert by_test["pp512"]["ngl"] == 99
    assert by_test["pp512"]["fa"] == 1


def test_parse_drops_separator_and_build_lines():
    rows = harness.parse_llama_bench(_BENCH_PLAIN)
    for r in rows:
        # No '---' separator row or 'build:' footer leaked in as a data row.
        assert r["test"] in ("pp512", "tg128")
        assert "build" not in str(r["model"]).lower()


def test_parse_empty_text_returns_empty_list():
    assert harness.parse_llama_bench("") == []


def test_parse_wide_depth_table_extracts_t_s():
    # The wider layout (extra n_batch/n_ubatch/type_k/type_v columns) must still parse:
    # test + t/s are always the last two columns.
    rows = harness.parse_llama_bench(_BENCH_DEPTH)
    assert harness.select_bench(rows, "pp512") == pytest.approx(1379.54)
    assert harness.select_bench(rows, "tg128") == pytest.approx(89.94)


# ===========================================================================
# PURE unit tests — select_bench. No GPU / server / model.
# ===========================================================================

def test_select_bench_finds_headline():
    rows = harness.parse_llama_bench(_BENCH_PLAIN)
    assert harness.select_bench(rows, "pp512") == pytest.approx(1154.63)
    assert harness.select_bench(rows, "tg128") == pytest.approx(79.91)


def test_select_bench_missing_test_returns_none():
    rows = harness.parse_llama_bench(_BENCH_PLAIN)
    assert harness.select_bench(rows, "nonexistent") is None


def test_select_bench_default_depth_ignores_depth_rows():
    # With both a headline (depth 0) and a '@ d16384' row present, the default
    # depth=0 selection picks the headline, never the deep-context number.
    rows = harness.parse_llama_bench(_BENCH_DEPTH)
    pp = harness.select_bench(rows, "pp512")  # depth=0 default
    assert pp == pytest.approx(1379.54)
    assert pp != pytest.approx(418.72)  # the d16384 value must not leak through


# ===========================================================================
# PURE unit tests — check_performance (the gate decision). No GPU.
# ===========================================================================

def test_check_performance_pass():
    rows = harness.parse_llama_bench(_BENCH_PLAIN)
    results = harness.check_performance(rows, _TH)
    by_name = {r.name: r for r in results}
    assert set(by_name) == {"pp512", "tg128"}
    assert by_name["pp512"].passed is True
    assert by_name["tg128"].passed is True
    assert by_name["pp512"].severity == "aggregate"


def test_check_performance_fail_below_floor():
    rows = harness.parse_llama_bench(_BENCH_SLOW)
    by_name = {r.name: r for r in harness.check_performance(rows, _TH)}
    assert by_name["pp512"].passed is False  # 1000 < 1097
    assert by_name["tg128"].passed is False  # 70 < 75.9


def test_check_performance_at_floor_passes():
    # Boundary: exactly at the floor passes (the gate is >=, not >).
    rows = harness.parse_llama_bench(_BENCH_PLAIN)
    th_at = {"pp512_min": 1154.63, "tg128_min": 79.91}
    by_name = {r.name: r for r in harness.check_performance(rows, th_at)}
    assert by_name["pp512"].passed is True
    assert by_name["tg128"].passed is True


def test_check_performance_missing_pp_fails():
    # A bench table missing pp512 -> select_bench None -> that check fails (no silent pass).
    rows = harness.parse_llama_bench(_BENCH_NO_PP)
    by_name = {r.name: r for r in harness.check_performance(rows, _TH)}
    assert by_name["pp512"].passed is False
    assert by_name["tg128"].passed is True


# ===========================================================================
# QUALITY GATE — gated behind --bench-file (a saved llama-bench table).
# Skips with no file, so bare `pytest` (no GPU) still runs the pure unit tests
# above. certify.sh is the only thing that produces --bench-file.
# ===========================================================================

def _opt(request, name, default=None):
    """request.config.getoption that degrades to `default` if the (frozen) option
    is not registered, so this file never errors when run outside certify.sh."""
    try:
        return request.config.getoption(name)
    except (ValueError, KeyError):
        return default


def _thresholds(request):
    spec_path = _opt(request, "--spec") or harness.SPEC_PATH
    spec = harness.load_spec(spec_path)
    return harness.resolve_thresholds(spec, perf_tolerance=_opt(request, "--perf-tolerance"))


def test_performance_gate(request):
    bench_file = _opt(request, "--bench-file")
    if not bench_file:
        pytest.skip("no --bench-file (saved llama-bench table); certify.sh produces it")
    if not os.path.isfile(bench_file):
        pytest.fail(f"--bench-file given but does not exist: {bench_file}")
    with open(bench_file, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    rows = harness.parse_llama_bench(text)
    assert rows, f"no llama-bench data rows parsed from {bench_file}"
    results = harness.check_performance(rows, _thresholds(request))
    failed = [r for r in results if not r.passed]
    assert not failed, "perf gate failed: " + "; ".join(f"{r.name}: {r.detail}" for r in failed)
