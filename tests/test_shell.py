"""Shell helpers that do not need llama.cpp or a GPU."""
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCRIPTS = [
    "env.sh",
    "tests/certify.sh",
    *sorted(str(p.relative_to(ROOT)) for p in (ROOT / "scripts").glob("*.sh")),
]


def _bash(script, env=None):
    merged = os.environ.copy()
    merged.pop("WORK", None)
    merged.pop("CONV_PY", None)
    merged.pop("HF_BIN", None)
    if env:
        merged.update(env)
    return subprocess.run(
        ["bash", "-lc", script],
        cwd=ROOT,
        env=merged,
        text=True,
        capture_output=True,
        check=False,
    )


def test_bash_syntax():
    for rel in SCRIPTS:
        proc = subprocess.run(
            ["bash", "-n", rel],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert proc.returncode == 0, f"{rel}: {proc.stderr}"


def test_work_defaults_to_repo(tmp_path):
    # Source with WORK unset and point the side-effect dirs at tmp via the default
    # only after checking the resolved path. HOME is set so oneAPI's absence is fine.
    proc = _bash(
        f'cd "{ROOT}" && unset WORK && source ./env.sh && printf "%s" "$WORK"',
        env={"HOME": str(tmp_path)},
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == str(ROOT)


def test_ngl_for_full_partial_and_missing(tmp_path):
    small = tmp_path / "small.gguf"
    big = tmp_path / "big.gguf"
    small.write_bytes(b"")
    # Sparse files: stat reports the logical size, no 6 GiB is written.
    subprocess.run(["truncate", "-s", "1M", str(small)], check=True)
    subprocess.run(["truncate", "-s", "6000M", str(big)], check=True)
    proc = _bash(
        f'''
set -euo pipefail
source "{ROOT}/env.sh"
echo "small=$(ngl_for "{small}")"
echo "big=$(ngl_for "{big}")"
echo "missing=$(ngl_for "{tmp_path}/missing.gguf")"
''',
        env={"HOME": str(tmp_path), "WORK": str(tmp_path / "work"), "VRAM_GIB": "8", "N_LAYERS": "48"},
    )
    assert proc.returncode == 0, proc.stderr
    lines = dict(line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line)
    assert lines["small"] == "99"
    # 6000 MiB, budget (8-6)*1024 = 2048 MiB, 6000/48 = 125 MiB/layer -> 16 layers.
    assert lines["big"] == "16"
    assert lines["missing"] == "99"
