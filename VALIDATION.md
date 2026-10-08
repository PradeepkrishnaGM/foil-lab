# Validation

The validation material of v1.3.1 (two reference files, `foil_lab.validation` and
`examples/validation_report.py`) was removed in milestone M1, together with the claims that rested
on it. The reasons are recorded in `DECISIONS.md` (M1, Q9).

Accuracy is now measured by the benchmark in `foil_lab/benchmark/`: the same coordinate file is
analysed by Foil-Lab and by a locally installed XFOIL, for six sections from Reynolds number 5e4 to
6e6. The reference polars are committed with their provenance under
`src/foil_lab/data/reference/xfoil/`, so the comparison runs without XFOIL.

- Measured state of v1.3.1: [`docs/baseline_v1.3.1.md`](docs/baseline_v1.3.1.md).
- Exact-solution and consistency tests: `tests/test_exact_solutions.py`.

This document will be generated from the benchmark output in a later milestone (M9).

## Reproducing

```bash
python -m foil_lab.benchmark all       # needs xfoil and xvfb-run for the reference step
python -m foil_lab.benchmark report    # comparison only, from the committed references
```
