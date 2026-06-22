"""pytest configuration for the QUEST certification suite.

Wires tests/lib onto sys.path[0] so `import harness` works, registers CLI
options + markers, and defines the GPU/server GATE: the server/model/bench/ppl
fixtures pytest.skip() unless the matching --option is given. So:

  * bare `pytest`              -> collects everything, RUNS only the pure unit tests
  * `pytest --collect-only`    -> shows the full suite (no GPU touched)
  * certify.sh passes the real --server-url/--model-path/--bench-file/--ppl-file
    to actually exercise the gated tests against a live B70 server it owns.
"""
import os
import sys
import json

import pytest

# --- sys.path wiring: tests/lib at sys.path[0] so `import harness` resolves ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib"))


# --------------------------------------------------------------------------- #
# CLI options
# --------------------------------------------------------------------------- #
def pytest_addoption(parser):
    g = parser.getgroup("quest-cert")
    g.addoption("--server-url", action="store", default=None,
                help="Base URL of a live llama-server (gates behavioral tests).")
    g.addoption("--model-path", action="store", default=None,
                help="Path to the GGUF under test (gates metadata tests; header-only read).")
    g.addoption("--ref-answers", action="store", default=None,
                help="Path to the BF16 reference answers JSON (gates agreement/certify).")
    g.addoption("--bench-file", action="store", default=None,
                help="Path to a saved llama-bench markdown table (gates performance test).")
    g.addoption("--ppl-file", action="store", default=None,
                help="Path to a saved llama-perplexity log (gates perplexity test).")
    g.addoption("--spec", action="store", default=None,
                help="Path to spec.yaml (default: harness.SPEC_PATH next to tests/).")
    g.addoption("--perf-tolerance", action="store", type=float, default=None,
                help="Fractional perf tolerance; None -> spec thresholds.perf.tolerance (0.05).")
    g.addoption("--out-dir", action="store", default=None,
                help="Directory for emitted artifacts (e.g. certify_<quant>.json).")


def pytest_configure(config):
    for name, desc in [
        ("scoring", "pure scorer unit tests (no GPU, synthetic data)"),
        ("agentic", "behavioral tool/reason probes (needs --server-url)"),
        ("metadata", "GGUF metadata/topology checks (needs --model-path)"),
        ("performance", "llama-bench perf gate (needs --bench-file)"),
        ("perplexity", "llama-perplexity quality gate (needs --ppl-file)"),
        ("gpu", "requires the GPU/server"),
    ]:
        config.addinivalue_line("markers", f"{name}: {desc}")


# --------------------------------------------------------------------------- #
# spec / thresholds (always available; pure, no GPU)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def spec(request):
    import harness
    path = request.config.getoption("--spec") or harness.SPEC_PATH
    return harness.load_spec(path)


@pytest.fixture(scope="session")
def perf_tolerance(request):
    return request.config.getoption("--perf-tolerance")


@pytest.fixture(scope="session")
def thresholds(spec, perf_tolerance):
    import harness
    return harness.resolve_thresholds(spec, perf_tolerance=perf_tolerance)


@pytest.fixture(scope="session")
def out_dir(request):
    return request.config.getoption("--out-dir")


# --------------------------------------------------------------------------- #
# GPU / server GATE — each fixture skips unless its option is supplied
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def server_url(request):
    url = request.config.getoption("--server-url")
    if not url:
        pytest.skip("no --server-url (live server gate): bare pytest runs pure tests only")
    return url


@pytest.fixture(scope="session")
def model_path(request):
    path = request.config.getoption("--model-path")
    if not path:
        pytest.skip("no --model-path (GGUF gate)")
    if not os.path.isfile(path):
        pytest.skip(f"--model-path not found: {path}")
    return path


@pytest.fixture(scope="session")
def ref_answers(request):
    path = request.config.getoption("--ref-answers")
    if not path:
        pytest.skip("no --ref-answers (BF16 reference gate)")
    if not os.path.isfile(path):
        pytest.skip(f"--ref-answers not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def bench_file(request):
    """Returns the CONTENTS of the saved llama-bench table (str), or skips."""
    path = request.config.getoption("--bench-file")
    if not path:
        pytest.skip("no --bench-file (performance gate)")
    if not os.path.isfile(path):
        pytest.skip(f"--bench-file not found: {path}")
    with open(path, "r") as f:
        return f.read()


@pytest.fixture(scope="session")
def ppl_file(request):
    """Returns the CONTENTS of the saved llama-perplexity log (str), or skips."""
    path = request.config.getoption("--ppl-file")
    if not path:
        pytest.skip("no --ppl-file (perplexity gate)")
    if not os.path.isfile(path):
        pytest.skip(f"--ppl-file not found: {path}")
    with open(path, "r") as f:
        return f.read()


@pytest.fixture(scope="session")
def cand_answers(server_url, spec):
    """Collect candidate answers from the live server (gated via server_url)."""
    import harness
    max_tokens = spec.get("meta", {}).get("max_tokens", 4096)
    return harness.collect(server_url, spec, max_tokens=max_tokens)
