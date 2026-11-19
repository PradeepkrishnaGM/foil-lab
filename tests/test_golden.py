"""Golden values: recorded before any speed optimisation and unchanged after it.

The file ``tests/data/golden/golden_v21.json`` is written by ``python -m tests.golden_cases`` with
single-threaded BLAS (multi-threaded BLAS changes the last digits; DECISIONS.md M7). The cases run
in a subprocess with the same setting, so the comparison is exact. The values are bit-identical on
the machine that recorded them; another CPU may differ in the last bits (slow test, not in CI).
A difference means the numbers changed: acceptable only as a recorded decision (DECISIONS.md), with
the file re-recorded in the same commit.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
GOLDEN = ROOT / "tests" / "data" / "golden" / "golden_v21.json"


@pytest.mark.slow
def test_golden_values_unchanged():
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    out = subprocess.run([sys.executable, "-m", "tests.golden_cases"], cwd=ROOT, env=env,
                         capture_output=True, text=True, timeout=1800, check=True).stdout
    want = json.loads(GOLDEN.read_text())
    got = json.loads(out)
    assert got.keys() == want.keys()
    for case in want:
        assert got[case] == want[case], case
