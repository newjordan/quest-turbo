"""CLI output for the offline grade and cert commands."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def _run(args):
    env = os.environ.copy()
    env["NO_COLOR"] = "1"
    env["PYTHONPATH"] = str(ROOT / "tests" / "lib")
    return subprocess.run(
        [PY, *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_grade_recorded_q5_ships():
    proc = _run([
        "scripts/04b_probes.py", "grade",
        "results/04b_answers", "BF16", "Q5_K_M",
    ])
    assert proc.returncode == 0, proc.stderr
    assert "Q5_K_M" in proc.stdout
    assert "SHIP" in proc.stdout
    assert "8/9" in proc.stdout
    assert "10/10" in proc.stdout
    # Columns line up with the header labels.
    header = next(line for line in proc.stdout.splitlines() if line.startswith("model"))
    assert "TOOL" in header and "REASON" in header and "THINK" in header and "verdict" in header


def test_harness_certify_cli_writes_json(tmp_path):
    dest = tmp_path / "report.json"
    proc = _run([
        str(ROOT / "tests" / "lib" / "harness.py"), "certify",
        "--cand", "results/04b_answers/Q5_K_M.json",
        "--ref", "results/04b_answers/BF16.json",
        "--json-out", str(dest),
    ])
    assert proc.returncode == 0, proc.stderr
    assert "SHIP" in proc.stdout
    report = json.loads(dest.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["sanity_ok"] is True


def test_harness_ppl_and_bench_cli():
    ppl = _run([str(ROOT / "tests" / "lib" / "harness.py"), "ppl", "results/04_ppl_Q5.txt"])
    assert ppl.returncode == 0, ppl.stderr
    assert "7.1905" in ppl.stdout
    assert "PASS" in ppl.stdout

    bench = _run([str(ROOT / "tests" / "lib" / "harness.py"), "bench", "results/05_depth.txt"])
    assert bench.returncode == 0, bench.stderr
    assert "1379.54" in bench.stdout or "1379.5" in bench.stdout
    assert "PASS" in bench.stdout


def test_grade_rejects_a_broken_candidate(tmp_path):
    src = json.loads((ROOT / "results" / "04b_answers" / "Q5_K_M.json").read_text(encoding="utf-8"))
    for rec in src["reason"].values():
        rec["norm"] = "0"
        rec["think_ok"] = False
    broken = tmp_path / "answers"
    broken.mkdir()
    (broken / "BF16.json").write_text(
        (ROOT / "results" / "04b_answers" / "BF16.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (broken / "BAD.json").write_text(json.dumps(src), encoding="utf-8")
    proc = _run(["scripts/04b_probes.py", "grade", str(broken), "BF16", "BAD"])
    assert proc.returncode == 1
    assert "NO-SHIP" in proc.stdout
