"""M7: the example notebook runs on a CPU in under a minute (SPEC.md 4.8; milestone M7)."""

import time
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "examples" / "tour.ipynb"


@pytest.mark.slow
def test_example_notebook_runs_in_under_a_minute(tmp_path):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    nb = nbformat.read(NOTEBOOK, as_version=4)
    t0 = time.perf_counter()
    nbclient.NotebookClient(nb, timeout=120, kernel_name="python3",
                            resources={"metadata": {"path": str(tmp_path)}}).execute()
    elapsed = time.perf_counter() - t0
    errors = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", [])
              if o.get("output_type") == "error"]
    assert not errors
    assert elapsed < 60.0, f"notebook took {elapsed:.1f} s"
