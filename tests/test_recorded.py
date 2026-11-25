"""Offline checks against the committed campaign logs.

These do not re-run the model. They lock the parsers to the logs that are
already in the tree and re-apply the ship rule to the saved probe answers.
"""
import json
from pathlib import Path

import pytest

import harness

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def test_recorded_q5_matches_bf16_ship_rule():
    cand = json.loads((RESULTS / "04b_answers" / "Q5_K_M.json").read_text(encoding="utf-8"))
    ref = json.loads((RESULTS / "04b_answers" / "BF16.json").read_text(encoding="utf-8"))
    report = harness.certify(cand, ref, harness.resolve_thresholds(harness.load_spec()))
    assert report.sanity_ok
    assert report.passed
    assert report.hard_fails == []
    assert report.summary["cand"]["tool_key"] == 8
    assert report.summary["cand"]["reason_key"] == 10
    assert report.summary["cand"]["think_ok"] == 10
    assert report.summary["ref"]["tool_key"] == 8
    assert report.summary["agreement"]["tool_agree"] == 8
    assert report.summary["agreement"]["reason_agree"] == 10


def test_recorded_ppl_log():
    value = harness.parse_ppl((RESULTS / "04_ppl_Q5.txt").read_text(encoding="utf-8", errors="replace"))
    assert value == pytest.approx(7.1905)
    assert harness.check_ppl(value, harness.resolve_thresholds(harness.load_spec())).passed


def test_recorded_depth_bench_kv_q8():
    rows = harness.parse_llama_bench((RESULTS / "05_depth.txt").read_text(encoding="utf-8"))
    assert harness.select_bench(rows, "pp512") == pytest.approx(1379.54)
    assert harness.select_bench(rows, "tg128") == pytest.approx(89.94)
    assert harness.select_bench(rows, "tg128", depth=16384) == pytest.approx(17.73)
    assert harness.select_bench(rows, "tg128", depth=32768) == pytest.approx(9.92)
    assert harness.select_bench(rows, "pp512", depth=32768) == pytest.approx(240.04)


def test_recorded_ubatch_prefill():
    rows = harness.parse_llama_bench((RESULTS / "05_ub.txt").read_text(encoding="utf-8"))
    pp = [row["t_s"] for row in rows if row["test"] == "pp4096"]
    assert pp == pytest.approx([1130.19, 1702.95, 1803.74])


def test_fa_sweep_has_two_headline_rows():
    rows = harness.parse_llama_bench((RESULTS / "05_fa.txt").read_text(encoding="utf-8"))
    dups = harness.repeated_headline_tests(rows)
    assert len(dups["pp512"]) == 2
    assert len(dups["tg128"]) == 2
    # First row is FA off. Callers that pass this file to the gate must notice.
    assert harness.select_bench(rows, "pp512") == pytest.approx(1596.60)
    results = harness.check_performance(rows, {"pp512_min": 1.0, "tg128_min": 1.0})
    assert any("first of 2" in c.detail for c in results)


def test_quant_size_table_uses_binary_gigabytes():
    text = (RESULTS / "03_quant_sizes.txt").read_text(encoding="utf-8")
    assert "21725581024" in text
    # The historical file labels the 1024^3 column "GB". 21725581024 / 1024**3 == 20.23.
    assert abs(21725581024 / 1024**3 - 20.23) < 0.01
    assert "18556686048" in text
    assert "25092531936" in text
