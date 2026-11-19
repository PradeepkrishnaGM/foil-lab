"""The validation document generator imports and builds without benchmark output (v2.2)."""

from pathlib import Path

from foil_lab.benchmark import validation


def test_build_without_benchmark_output_states_no_flap_claim(tmp_path: Path) -> None:
    md = validation.build(tmp_path / "main", tmp_path / "flaps")
    assert md.startswith("# Validation")
    assert "## Limits" in md
    assert "- Plain flaps are not validated; all deflections are reported without a claim." in md
