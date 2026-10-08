"""IO round-trip tests."""
import numpy as np
from pathlib import Path
from foil_lab.utils.io import load_dat, save_dat
from foil_lab.generators import generate_naca4


def test_save_load_roundtrip(tmp_path):
    af = generate_naca4("2412", 60)
    fp = tmp_path / "test.dat"
    save_dat(fp, af, header="Test")
    loaded = load_dat(fp)
    assert loaded.shape == af.shape
    assert np.allclose(loaded, af, atol=1e-5)
