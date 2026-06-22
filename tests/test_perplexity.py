"""QUALITY GATE — perplexity (Q5 wikitext gross-breakage guard).

AUTHORITATIVE DESIGN (frozen contract): this test file NEVER touches the B70 GPU and
NEVER starts llama-perplexity. The GPU run happens exactly once, in tests/certify.sh,
which executes

    $RUN_BIN/llama-perplexity -m <quant>.gguf -ngl <ngl_for> -f <wikitext> --chunks 40
        2>&1 | tee $RES/certify_ppl_<quant>.log

and passes that saved log to pytest as --ppl-file. Here we ONLY parse that log via
harness.parse_ppl and assert the wikitext PPL is within the spec ceiling
(thresholds.ppl.max == 8.5) via harness.check_ppl. The Q5 anchor is ~7.19.

Two tiers live in this file:
  * PURE unit tests — feed synthetic llama-perplexity text to the import-pure parser /
    gate functions. These RUN under bare `pytest` with no server, no model, no GPU, so
    the PPL parsing + the ship/no-ship decision are CI-verifiable.
  * The gated quality test — skips unless --ppl-file is given; then it is a pure parse +
    threshold check of the saved log.
"""
import os

import pytest

import harness


# ===========================================================================
# PURE unit tests — synthetic llama-perplexity text. No GPU / server / model.
# ===========================================================================

# Verbatim shape of the tail of results/04_ppl_Q5.txt (Q5_K_M, wikitext, 40 chunks).
_REAL_PPL_LOG = """\
0.23.964.873 I perplexity: calculating perplexity over 40 chunks, n_ctx=512, batch_size=2048, n_seq=4
0.26.384.331 I perplexity: 2.42 seconds per pass - ETA 0.40 minutes
[1]5.3976,[2]7.7320,[3]7.1532,[38]7.1465,[39]7.1432,[40]7.1905,
0.41.889.689 I Final estimate: PPL = 7.1905 +/- 0.20116
"""


def test_parse_ppl_real_tail():
    # The exact line certify.sh's llama-perplexity log ends with.
    assert harness.parse_ppl(_REAL_PPL_LOG) == pytest.approx(7.1905)


def test_parse_ppl_minimal_line():
    assert harness.parse_ppl("Final estimate: PPL = 7.19 +/- 0.2") == pytest.approx(7.19)


def test_parse_ppl_tolerates_extra_whitespace():
    # llama-perplexity spacing around '=' varies; the parser must absorb it.
    assert harness.parse_ppl("Final estimate:  PPL =   8.4321 +/- 0.1") == pytest.approx(8.4321)


def test_parse_ppl_integer_value():
    assert harness.parse_ppl("Final estimate: PPL = 9") == pytest.approx(9.0)


def test_parse_ppl_picks_the_estimate_not_a_chunk_number():
    # Per-chunk readouts like [40]7.1905 must NOT be mistaken for the final estimate.
    assert harness.parse_ppl("[40]999.0000,\nFinal estimate: PPL = 7.1905 +/- 0.2") == pytest.approx(7.1905)


def test_parse_ppl_no_estimate_returns_none():
    assert harness.parse_ppl("perplexity: tokenizing the input ..\nno final line here") is None


def test_parse_ppl_empty_returns_none():
    assert harness.parse_ppl("") is None


# --- the gate decision (check_ppl) on synthetic numbers --------------------

# Minimal thresholds shape matching resolve_thresholds() output: thresholds['ppl']['max'].
_TH = {"ppl": {"max": 8.5, "anchor": 7.19}}


def test_check_ppl_pass_under_ceiling():
    r = harness.check_ppl(7.1905, _TH)
    assert r.passed is True
    assert r.name == "ppl"
    assert r.severity == "aggregate"


def test_check_ppl_q5_anchor_passes_with_margin():
    # The recorded Q5 wikitext anchor (7.19) must clear the 8.5 gate comfortably.
    assert harness.check_ppl(_TH["ppl"]["anchor"], _TH).passed is True


def test_check_ppl_at_ceiling_passes():
    # Boundary: ppl == max is acceptable (the gate is <=, not <).
    assert harness.check_ppl(8.5, _TH).passed is True


def test_check_ppl_over_ceiling_fails():
    assert harness.check_ppl(8.6, _TH).passed is False


def test_check_ppl_none_fails():
    # An unparseable log (ppl is None) must NOT silently pass the gate.
    assert harness.check_ppl(None, _TH).passed is False


# ===========================================================================
# QUALITY GATE — gated behind --ppl-file (a saved llama-perplexity log).
# Skips with no file, so bare `pytest` (no GPU) still collects + runs the pure
# unit tests above. certify.sh is the only thing that produces --ppl-file.
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


def test_perplexity_gate(request):
    ppl_file = _opt(request, "--ppl-file")
    if not ppl_file:
        pytest.skip("no --ppl-file (saved llama-perplexity log); certify.sh produces it")
    if not os.path.isfile(ppl_file):
        pytest.fail(f"--ppl-file given but does not exist: {ppl_file}")
    with open(ppl_file, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    ppl = harness.parse_ppl(text)
    assert ppl is not None, f"no 'Final estimate: PPL = ...' line in {ppl_file}"
    result = harness.check_ppl(ppl, _thresholds(request))
    assert result.passed, f"PPL gate failed: {result.detail}"
