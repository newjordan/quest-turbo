"""GGUF topology certification for QUEST-30B-RL.

Reads the GGUF metadata (KV header only — header-only mmap read; does NOT load
tensors or touch the GPU) via harness.read_gguf_metadata and asserts the frozen
qwen3moe topology fields against spec['metadata'].

GATED: every test here depends on the `model_path` fixture, which pytest.skip()s
unless --model-path is supplied. So `pytest` with no model collects these tests
but skips them; `pytest --collect-only` lists them.

Ground truth (= tests/spec.yaml metadata): arch=qwen3moe, 48 layers, 128 experts,
8 used, 32 attn heads, 4 KV heads, ctx 262144, embd 2048, ffn 6144,
expert_ffn 768, rope freq_base ~1e7.
"""
import pytest

import harness

# Loaded at collection time so each metadata field becomes its own parametrized test.
try:
    _META = harness.load_spec()["metadata"]
except Exception:  # spec.yaml not landed yet -> collect as an empty (skipped) param set
    _META = []


@pytest.fixture(scope="session")
def gguf_meta(model_path):
    """Header-only KV read of the model file. Gated through `model_path` (skips
    without --model-path); never reads tensor data and never opens the GPU."""
    return harness.read_gguf_metadata(model_path)


@pytest.mark.parametrize("exp", _META, ids=[e["key"] for e in _META])
def test_metadata_field(exp, gguf_meta):
    key = exp["key"]
    assert key in gguf_meta, f"GGUF missing metadata key: {key}"
    actual = gguf_meta[key]
    expected = exp["expected"]
    typ = exp["type"]
    if typ == "int":
        assert int(actual) == int(expected), f"{key}: {actual} != {expected}"
    elif typ == "str":
        assert str(actual) == str(expected), f"{key}: {actual!r} != {expected!r}"
    elif typ == "float":
        rel = float(exp.get("rel_tol", 1e-6))
        assert abs(float(actual) - float(expected)) <= rel * abs(float(expected)), (
            f"{key}: {actual} not within rel_tol={rel} of {expected}"
        )
    else:
        pytest.fail(f"unknown metadata type {typ!r} for {key}")


def test_metadata_all_via_helper(gguf_meta):
    """Roll-up using harness.check_metadata so the helper itself is exercised."""
    results = harness.check_metadata(gguf_meta, _META)
    failed = [c for c in results if not c.passed]
    assert not failed, "GGUF metadata mismatches: " + "; ".join(
        f"{c.name}={c.detail}" for c in failed
    )


def test_architecture_is_qwen3moe(gguf_meta):
    """Spot-check the single most load-bearing field independent of the spec loop."""
    assert str(gguf_meta.get("general.architecture")) == "qwen3moe"
