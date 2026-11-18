# Testing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
python -m pytest tests/ -q -m "not slow"     # fast suite: about 470 tests, under a minute
python -m pytest tests/ -q                   # everything, including viscous runs: about 30 minutes
QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui.py -q   # GUI without a screen
```

- Tests marked `slow` take more than about 2 seconds each (viscous solves, the notebook, the
  process-pool batch run).
- Tests marked `xfoil` call the external XFOIL program and are skipped when `xfoil` or `xvfb-run`
  is not on the PATH. All other benchmark tests read the committed reference polars and run
  without XFOIL.
- Determinism: identical input gives bit-identical output (`test_deterministic_inviscid_and_viscous`).

The XFOIL benchmark itself (reference generation and the full comparison) is run with
`python -m foil_lab.benchmark`; see VALIDATION.md, section "Reproducing".
