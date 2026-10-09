"""Boundary-layer diagnostic against XFOIL (M3). Slow; the regeneration test needs XFOIL.

These tests check that the comparison runs and that its inputs are the ones recorded; they do
not assert accuracy (that is the M3 benchmark).
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from tests.diagnostics import bl_compare as bc


@pytest.mark.slow
@pytest.mark.parametrize("alpha", bc.ANGLES)
def test_committed_dump_matches_canonical_geometry(alpha):
    from foil_lab.benchmark.xfoil import write_coordinate_file
    with tempfile.TemporaryDirectory() as d:
        sha = write_coordinate_file(Path(d) / "n.dat", bc.canonical_coords(), "NACA2412")
    assert bc.provenance_sha(bc.DATA / bc.dump_name(alpha)) == sha


@pytest.mark.slow
@pytest.mark.parametrize("alpha", bc.ANGLES)
def test_comparison_runs_without_xfoil(alpha):
    c = bc.compare(alpha)
    for lab in ("upper", "lower"):
        rows = c["tables"][lab]
        assert len(rows) == len(bc.STATIONS)
        assert all(np.isfinite(r["dstar_xf"]) and np.isfinite(r["dstar_cpl"]) for r in rows)
    assert np.isfinite(c["cl_core_with_xfoil_dstar"])


@pytest.mark.slow
@pytest.mark.xfoil
def test_regenerated_dumps_match_committed():
    """XFOIL is deterministic: a fresh dump equals the committed one (data rows)."""
    with tempfile.TemporaryDirectory() as d:
        fresh = bc.run_xfoil_dumps(Path(d))
        for a, path in fresh.items():
            new, _ = bc.read_dump(path)
            old, _ = bc.read_dump(bc.DATA / bc.dump_name(a))
            assert new.shape == old.shape and np.allclose(new, old, equal_nan=True, atol=0, rtol=0)
