"""Header-only GGUF read against a tiny synthetic file, not a model download."""
from gguf import GGUFWriter

import harness


def test_read_gguf_metadata_roundtrip(tmp_path):
    path = tmp_path / "tiny.gguf"
    writer = GGUFWriter(str(path), arch="qwen3moe")
    writer.add_name("tiny")
    writer.add_block_count(48)
    writer.add_expert_count(128)
    writer.add_context_length(262144)
    writer.add_rope_freq_base(1.0e7)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    meta = harness.read_gguf_metadata(path)
    expectations = [
        {"key": "general.architecture", "expected": "qwen3moe", "type": "str"},
        {"key": "qwen3moe.block_count", "expected": 48, "type": "int"},
        {"key": "qwen3moe.expert_count", "expected": 128, "type": "int"},
        {"key": "qwen3moe.context_length", "expected": 262144, "type": "int"},
        {"key": "qwen3moe.rope.freq_base", "expected": 1.0e7, "type": "float", "rel_tol": 1.0e-6},
    ]
    failed = [c for c in harness.check_metadata(meta, expectations) if not c.passed]
    assert not failed, failed
    assert "qwen3moe.block_count" in meta
    assert isinstance(meta["qwen3moe.block_count"], int)
