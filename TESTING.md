# TESTING.md — How to test Foil Lab on your machine

This guide gets Foil Lab installed on a clean Fedora 43 box and tells you exactly
what to run and **what to send back** so the project's real-world status can be
assessed. Copy/paste each block; capture the output.

The whole point: our development happened in a container. This verifies it works
on real hardware, and surfaces install/environment problems we can't see.

---

## 0. System prerequisites (Fedora 43)

```bash
# Python 3.10+ and venv (Fedora 43 ships Python 3.13 — fine, we support >=3.10)
python3 --version
sudo dnf install -y python3-pip python3-virtualenv git
```

> If `python3 --version` is below 3.10, tell me — that's important.

---

## 1. Get the code and create a clean environment

```bash
cd ~                      # or wherever you keep projects
# (unzip the foil-lab-vX.Y.zip I gave you here, so you have ~/foil-lab)
cd foil-lab

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

---

## 2. Install — do BOTH, in order, and report each

### 2a. Minimal core install (numpy + scipy + matplotlib only)
```bash
pip install -e .
foillab info
```
**SEND ME:** the full output of both commands, including any errors. The `foillab
info` block should print and show `JAX: no`, `PySide6: no` — that's expected for
the minimal install.

### 2b. Full install (all optional backends)
```bash
pip install -e ".[all]"
foillab info
```
**SEND ME:** the output. Now JAX / PySide6 / pyarrow should say `yes`. If any
fail to install (JAX and PySide6 are the usual suspects on Linux), copy the pip
error — that's exactly the kind of thing we need to know.

---

## 3. Run the test suite

```bash
# from the foil-lab directory, with .venv active
QT_QPA_PLATFORM=offscreen python -m pytest tests/ -v 2>&1 | tee test_output.txt
```
**SEND ME:** the file `test_output.txt` (or the last ~40 lines + the final
summary line). 

What to expect on a healthy full install:
- A summary like `86 passed, 2 xfailed` (the 2 xfails are EXPECTED — they
  document known-incomplete low-Re high-alpha cases).
- **0 failed.** Any line with `FAILED` is a real problem — send it.
- `xpassed` (unexpected pass) on a VII test would actually be *good news* (means
  something converges better on your machine than ours) — flag it.

---

## 4. Functional smoke tests (the headline capabilities)

Run each and send the output. These exercise the real solvers.

### 4a. Inviscid + high-Re viscous (the validated core)
```bash
foillab polar --naca 2412 --start -4 --end 12 --step 2 --re 3000000 --vii transpiration
```
**Expect:** Cl rising ~linearly then bending; Cd a few thousandths rising with
alpha; `conv` in the status column for most low/moderate angles. Zero-lift around
alpha = -2 deg.

### 4b. The v1.3 headline — low-Re Newton VII
```bash
foillab polar --naca 2412 --start 0 --end 6 --step 1 --re 50000 --vii newton
```
**Expect (honest):** SOME angles `conv`, some `NOCONV`. Converged points should
have Cd around 0.015–0.025. This is the experimental low-Re path — partial
convergence is the known, documented state, NOT a bug. **SEND the table**; which
angles converge on your machine is genuinely useful data.

### 4c. Benchmark report (compares to committed XFOIL reference polars)
```bash
python -m foil_lab.benchmark run && python -m foil_lab.benchmark report
```
Writes `benchmark_output/report.md` and plots. See `VALIDATION.md`.

### 4d. JAX core (only if 2b installed JAX successfully)
```bash
python examples/jax_demo.py
```
**Expect:** a "~100x faster" batch-speedup line and exact-gradient numbers. SEND
the output. If JAX failed to install, skip this and just tell me.

### 4e. GUI (only if PySide6 installed; needs a desktop session)
```bash
python main.py          # or:  foillab-gui
```
**Expect:** a window with tabs (Geometry / Conditions / Run) and plot panels.
Try: type `2412`, click "Generate NACA", set Re=1000000, click "Run Case".
**SEND ME:** does it launch? Does a plot appear? A screenshot is ideal. If it
errors in the terminal, copy that.

---

## 5. Performance note (optional but useful)

```bash
python examples/batch_sweep.py
```
Tells me roughly how fast your machine runs a 500-case sweep. SEND the timing
lines if it completes.

---

## What to send me, condensed

1. `foillab info` output for BOTH installs (step 2).
2. Any pip install errors (especially JAX / PySide6).
3. `test_output.txt` (step 3) — at minimum the final summary + any FAILED lines.
4. The two polar tables (4a, 4b).
5. `benchmark_output/report.md` (4c).
6. JAX demo output (4d) if available.
7. GUI: launches? screenshot or error (4e).
8. Your `python3 --version` and whether you're on X11 or Wayland (for the GUI).

With that, I can tell you precisely: what installed, what works on real hardware
vs. only in our container, whether the low-Re Newton result reproduces outside
our environment, and where the real rough edges are. That's the honest
"how far has this actually progressed" assessment.
```
