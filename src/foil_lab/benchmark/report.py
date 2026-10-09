"""Compare benchmark results with the XFOIL references; write tables, plots and the report.

Definitions (DECISIONS.md, M1 Q5, Q6, D2, E2):

* Reference: XFOIL at the first ``BENCHMARK.pane_nodes`` value (160). The second value (240)
  gives the paneling sensitivity (160 vs 240).
* Attached-flow filter: requested angles from ``alpha_min`` up to, but excluding, the first
  positive angle at which the reference Cl reaches ``attached_fraction`` (0.9) times its maximum
  over the converged positive-alpha points. If no positive angle converged, the filter is empty.
* A point enters the error metrics only if XFOIL converged there and the method converged.
  Points the method did not converge on are counted next to the metrics, never dropped silently.
* Sections with a trailing-edge gap of zero are "sharp"; the accuracy targets are judged on the
  other ("open") sections only.
"""

from __future__ import annotations

import csv
import datetime as _dt
import math
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..params import BENCHMARK, SWEEP, BenchmarkParams
from .baseline_v131 import case_file, read_case
from .references import load_reference
from .xfoil import XfoilPolar

METHODS = ("inviscid", "oneway", "vii", "newton")
METHOD_LABEL = {"inviscid": "inviscid", "oneway": "one-way (default)",
                "vii": "transpiration VII", "newton": "Newton VII"}
PROVISIONAL = {"cl": 0.03, "cd_rel": 0.10, "cm": 0.01}


@dataclass
class MethodStats:
    """Comparison of one method with the reference on one case."""

    n_compared: int = 0          # filter angles where XFOIL converged and the method was run
    n_unconverged: int = 0       # ... of which the method did not converge (excluded)
    dcl: list[float] = field(default_factory=list)
    dcm: list[float] = field(default_factory=list)
    dcd_rel: list[float] = field(default_factory=list)
    dxtr_top: list[float] = field(default_factory=list)
    dxtr_bot: list[float] = field(default_factory=list)
    runtime: list[float] = field(default_factory=list)


@dataclass
class CaseResult:
    """All comparisons for one (section, Reynolds number)."""

    section: str
    reynolds: float
    sharp: bool
    filter_alphas: list[float]
    alpha_09: float | None
    xfoil_unconv_160: int
    xfoil_unconv_240: int
    recovered_by_240: int
    sens: MethodStats
    methods: dict[str, MethodStats]
    xfoil_runtime_s: float
    retried_160: int


def _mean(v: list[float]) -> float:
    return float(np.mean(v)) if v else math.nan


def _max(v: list[float]) -> float:
    return float(np.max(v)) if v else math.nan


def attached_filter(ref: XfoilPolar, bench: BenchmarkParams = BENCHMARK
                    ) -> tuple[list[float], float | None]:
    """Requested angles inside the attached-flow filter, and the cut-off angle."""
    pos = [(a, r.cl) for a, r in ref.rows.items() if r is not None and a >= 0.0]
    if not pos:
        return [], None
    clmax = max(cl for _, cl in pos)
    cut = next(a for a, cl in sorted(pos) if cl >= bench.attached_fraction * clmax)
    return [a for a in ref.alphas if SWEEP.alpha_min <= a < cut], cut


def _xfoil_runtime(ref: XfoilPolar) -> float:
    total = 0.0
    for part in ref.provenance.get("runs", "").split(";"):
        if part.strip().endswith(" s"):
            total += float(part.split(",")[-1].strip()[:-2])
    return total


def compare_case(section: str, reynolds: float, output_root: Path,
                 bench: BenchmarkParams = BENCHMARK) -> CaseResult:
    """Compare every v1.3.1 method with XFOIL for one case."""
    n_ref, n_alt = bench.pane_nodes
    ref = load_reference(section, reynolds, n_ref)
    alt = load_reference(section, reynolds, n_alt)
    filt, cut = attached_filter(ref, bench)
    un160 = sum(ref.rows[a] is None for a in filt)
    un240 = sum(alt.rows.get(a) is None for a in filt)
    recovered = sum(ref.rows[a] is None and alt.rows.get(a) is not None for a in filt)
    sens = MethodStats()
    for a in filt:
        r, s = ref.rows[a], alt.rows.get(a)
        if r is not None and s is not None:
            sens.n_compared += 1
            sens.dcl.append(abs(s.cl - r.cl))
            sens.dcm.append(abs(s.cm - r.cm))
            sens.dcd_rel.append(abs(s.cd - r.cd) / r.cd)
    rows = []
    for re_file in (reynolds, 0.0):
        path = case_file(output_root, section, re_file)
        if path.exists():
            rows += [r for r in read_case(path) if (r["method"] == "inviscid") == (re_file == 0.0)]
    methods: dict[str, MethodStats] = {}
    for m in METHODS:
        mrows = {r["alpha"]: r for r in rows if r["method"] == m}
        if not mrows:
            continue
        st = MethodStats()
        for a in filt:
            r, x = mrows.get(a), ref.rows[a]
            if r is None or x is None:
                continue
            st.n_compared += 1
            st.runtime.append(r["runtime_s"])
            if r["converged"] == "0" or not math.isfinite(r["cl"]):
                st.n_unconverged += 1
                continue
            st.dcl.append(abs(r["cl"] - x.cl))
            st.dcm.append(abs(r["cm"] - x.cm))
            if m != "inviscid":
                st.dcd_rel.append(abs(r["cd"] - x.cd) / x.cd)
                if math.isfinite(r["xtr_top"]):
                    st.dxtr_top.append(abs(r["xtr_top"] - x.xtr_top))
                    st.dxtr_bot.append(abs(r["xtr_bot"] - x.xtr_bot))
        methods[m] = st
    sharp = float(ref.provenance.get("trailing_edge_gap", "nan")) == 0.0
    return CaseResult(section, reynolds, sharp, filt, cut, un160, un240, recovered, sens, methods,
                      _xfoil_runtime(ref), len(ref.retried))


# --------------------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------------------- #
def _f(v: float, fmt: str) -> str:
    return "-" if not math.isfinite(v) else format(v, fmt)


def _pct(v: float) -> str:
    return "-" if not math.isfinite(v) else f"{100 * v:.1f} %"


def _pool(cases: list[CaseResult], method: str) -> MethodStats:
    out = MethodStats()
    for c in cases:
        st = c.sens if method == "sens" else c.methods.get(method)
        if st is None:
            continue
        out.n_compared += st.n_compared
        out.n_unconverged += st.n_unconverged
        for k in ("dcl", "dcm", "dcd_rel", "dxtr_top", "dxtr_bot", "runtime"):
            getattr(out, k).extend(getattr(st, k))
    return out


def _stats_cells(st: MethodStats, with_cd: bool = True) -> list[str]:
    return [str(st.n_compared), str(st.n_unconverged), str(len(st.dcl)),
            _f(_mean(st.dcl), ".4f"), _f(_max(st.dcl), ".4f"),
            _pct(_mean(st.dcd_rel)) if with_cd else "-", _pct(_max(st.dcd_rel)) if with_cd else "-",
            _f(_mean(st.dcm), ".4f"), _f(_max(st.dcm), ".4f")]


STAT_HEAD = ["compared", "not converged (excluded)", "used", "mean dCl", "max dCl",
             "mean rel dCd", "max rel dCd", "mean dCm", "max dCm"]


def _md(head: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_markdown(cases: list[CaseResult], bench: BenchmarkParams, plots: bool) -> str:
    """The baseline report as Markdown (with plot links if ``plots``)."""
    ref0 = load_reference(cases[0].section, cases[0].reynolds, bench.pane_nodes[0])
    res_valid = set(bench.reynolds_validated)
    lines = [
        "# Baseline v1.3.1 against XFOIL",
        "",
        f"Measured {_dt.date.today().isoformat()} on commit `{_git('rev-parse', '--short', 'HEAD')}` "
        "of branch `worktree-m1-benchmark`. Solver numerics are those of commit `be54d0a` "
        "(v1.3.1): M1 changed no solver code. Machine: "
        f"{platform.processor() or platform.machine()}, Python {platform.python_version()}, "
        "one BLAS thread per worker process.",
        "",
        f"Reference: XFOIL {ref0.provenance.get('xfoil_version')} "
        f"(`{ref0.provenance.get('xfoil_package')}`, run as `{ref0.provenance.get('xfoil_command')}`), "
        f"{bench.pane_nodes[0]} `PANE` nodes, N_crit 9, free transition, Mach 0, `ITER 200`, one retry "
        "per skipped angle. Both codes read the same canonical coordinate file (160 panels; NACA "
        "sections with the open trailing edge of NACA Report 460).",
        "",
        "Methods of v1.3.1: inviscid (`solve_flow`), one-way (`solve_flow` with Re, the default "
        "viscous path), transpiration VII (`solve_vii`, 200 iterations, tol 1e-3, e^n, N_crit 9), "
        "Newton VII (`solve_vii_newton`, 60 iterations, tol 1e-4) on a small subset.",
        "",
        "Rules: alpha -4 to 20 deg in 1 deg steps. Attached-flow filter: from -4 deg up to, not "
        "including, the first positive angle where XFOIL's Cl reaches 0.9 x its maximum. A point is "
        "used only if XFOIL converged and the method converged; points the method did not converge "
        "on are counted in 'not converged (excluded)'. Inviscid has no drag. 'Paneling sensitivity "
        "(160 vs 240)' is the difference between the two XFOIL runs; for the sharp-edged section "
        "(E387, gap 0) it is a lower bound, because XFOIL's result there keeps changing with node "
        "count. Accuracy targets are judged on open-trailing-edge sections only.",
        "",
        "The one-way path of v1.3.1 adds drag to the inviscid solution without changing it, so its "
        "Cl and Cm errors are identical to the inviscid ones (in the plots the inviscid curve lies "
        "under the one-way curve). One-way and inviscid results have no convergence criterion in "
        "v1.3.1 and are counted as converged.",
        "",
        f"Provisional targets (SPEC.md 4.1, validated range, attached flow): mean |dCl| <= "
        f"{PROVISIONAL['cl']}, mean relative |dCd| <= {PROVISIONAL['cd_rel']:.0%}, mean |dCm| <= "
        f"{PROVISIONAL['cm']}.",
        "",
    ]
    for group, sharp in (("open trailing edge (NACA 0012, 2412, 4412, 23012, Clark Y, "
                          "E387 with a 0.25 % gap)", False),
                         ("sharp trailing edge (E387)", True)):
        lines += [f"## Summary by Reynolds number: {group}", ""]
        rows = []
        for re in sorted({c.reynolds for c in cases}):
            sub = [c for c in cases if c.reynolds == re and c.sharp == sharp]
            if not sub:
                continue
            tag = f"{re:g}" + ("" if re in res_valid else " (not validated)")
            for m in ("sens",) + METHODS:
                st = _pool(sub, m)
                if st.n_compared == 0:
                    continue
                label = "XFOIL 240 vs 160 (paneling sensitivity)" if m == "sens" else METHOD_LABEL[m]
                rows.append([tag, label] + _stats_cells(st, m != "inviscid"))
        lines += [_md(["Re", "method"] + STAT_HEAD, rows), ""]
        valid = [c for c in cases if c.reynolds in res_valid and c.sharp == sharp]
        rows = []
        for m in ("sens",) + METHODS:
            st = _pool(valid, m)
            if st.n_compared == 0:
                continue
            met = []
            if m not in ("sens", "inviscid"):
                met = [("Cl", _mean(st.dcl) <= PROVISIONAL["cl"]),
                       ("Cd", _mean(st.dcd_rel) <= PROVISIONAL["cd_rel"]),
                       ("Cm", _mean(st.dcm) <= PROVISIONAL["cm"])]
            label = "XFOIL 240 vs 160 (paneling sensitivity)" if m == "sens" else METHOD_LABEL[m]
            verdict = ", ".join(f"{k} {'met' if ok else 'not met'}" for k, ok in met) or "-"
            rows.append([label] + _stats_cells(st, m != "inviscid") + [verdict])
        lines += [f"Validated range pooled (Re 5e5 to 6e6), {group}:", "",
                  _md(["method"] + STAT_HEAD + ["provisional targets"], rows), ""]
    lines += ["## XFOIL convergence inside the attached-flow filter", "",
              _md(["section", "Re", "filter angles", "alpha cut-off (deg)",
                   "XFOIL not converged at 160", "XFOIL not converged at 240",
                   "converged at 240 but not 160", "retried at 160 (any angle)",
                   "XFOIL run time 160 (s)"],
                  [[c.section, f"{c.reynolds:g}", str(len(c.filter_alphas)),
                    _f(c.alpha_09 if c.alpha_09 is not None else math.nan, "g"),
                    str(c.xfoil_unconv_160), str(c.xfoil_unconv_240), str(c.recovered_by_240),
                    str(c.retried_160), f"{c.xfoil_runtime_s:.1f}"] for c in cases]), ""]
    lines += ["## Per case and method", "",
              "Transition: mean |dx_tr/c| on upper / lower surface against XFOIL, over used points.",
              ""]
    rows = []
    for c in cases:
        for m in ("sens",) + METHODS:
            st = c.sens if m == "sens" else c.methods.get(m)
            if st is None or (m != "sens" and st.n_compared == 0 and not st.runtime):
                continue
            label = "XFOIL 240 vs 160" if m == "sens" else METHOD_LABEL[m]
            xtr = (f"{_f(_mean(st.dxtr_top), '.3f')} / {_f(_mean(st.dxtr_bot), '.3f')}"
                   if m in ("oneway", "vii", "newton") else "-")
            rt = _f(_mean(st.runtime), ".2f") if m != "sens" else "-"
            rows.append([c.section, f"{c.reynolds:g}", label] + _stats_cells(st, m != "inviscid")
                        + [xtr, rt])
    lines += [_md(["section", "Re", "method"] + STAT_HEAD + ["mean dx_tr up / low",
                                                               "mean run time per point (s)"], rows), ""]
    if plots:
        lines += ["## Plots", ""] + [f"- [{c.section} Re {c.reynolds:g}](plots/{c.section}_Re"
                                     f"{int(round(c.reynolds))}.png)" for c in cases] + [""]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------------------- #
def plot_case(c: CaseResult, output_root: Path, bench: BenchmarkParams) -> Path:
    """Cl-alpha, Cl-Cd and Cm-alpha overlays for one case; returns the PNG path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    for n, style in zip(bench.pane_nodes, (dict(color="k", marker="o", ms=3, lw=1.2),
                                           dict(color="0.55", ls="--", lw=1.0))):
        ref = load_reference(c.section, c.reynolds, n)
        a = [x for x in ref.alphas if ref.rows[x] is not None]
        cl = [ref.rows[x].cl for x in a]
        axs[0].plot(a, cl, label=f"XFOIL {n}", **style)
        axs[1].plot([ref.rows[x].cd for x in a], cl, label=f"XFOIL {n}", **style)
        axs[2].plot(a, [ref.rows[x].cm for x in a], label=f"XFOIL {n}", **style)
    rows = []
    for re_file in (c.reynolds, 0.0):
        p = case_file(output_root, c.section, re_file)
        if p.exists():
            rows += [r for r in read_case(p) if (r["method"] == "inviscid") == (re_file == 0.0)]
    colors = {"inviscid": "tab:blue", "oneway": "tab:orange", "vii": "tab:green", "newton": "tab:red"}
    for m in METHODS:
        mr = sorted((r for r in rows if r["method"] == m), key=lambda r: r["alpha"])
        if not mr:
            continue
        for conv in (True, False):
            sel = [r for r in mr if (r["converged"] != "0") == conv and math.isfinite(r["cl"])]
            if not sel:
                continue
            kw = dict(color=colors[m], marker="s" if conv else "x", ms=4,
                      ls="-" if conv else "none", lw=1,
                      label=METHOD_LABEL[m] + ("" if conv else " (not converged)"))
            al = [r["alpha"] for r in sel]
            axs[0].plot(al, [r["cl"] for r in sel], **kw)
            if m != "inviscid":
                axs[1].plot([r["cd"] for r in sel], [r["cl"] for r in sel], **kw)
            axs[2].plot(al, [r["cm"] for r in sel], **kw)
    if c.filter_alphas:
        for ax in (axs[0], axs[2]):
            ax.axvspan(min(c.filter_alphas), max(c.filter_alphas), color="0.92", zorder=0)
    axs[0].set(xlabel="alpha (deg)", ylabel="Cl")
    axs[1].set(xlabel="Cd", ylabel="Cl")
    axs[2].set(xlabel="alpha (deg)", ylabel="Cm c/4")
    for ax in axs:
        ax.grid(alpha=0.3)
    axs[1].set_xlim(0.0, 0.03)  # readable attached range; larger drag values fall outside
    axs[0].legend(fontsize=7)
    fig.suptitle(f"{c.section}, Re {c.reynolds:g}: v1.3.1 against XFOIL "
                 f"(shaded: attached-flow filter)")
    fig.tight_layout()
    out = output_root / "plots" / f"{c.section}_Re{int(round(c.reynolds))}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return out


def _tables_csv(cases: list[CaseResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["section", "reynolds", "sharp_te", "method"] + [h.replace(" ", "_") for h in STAT_HEAD])
        for c in cases:
            for m in ("sens",) + METHODS:
                st = c.sens if m == "sens" else c.methods.get(m)
                if st is None:
                    continue
                w.writerow([c.section, f"{c.reynolds:g}", int(c.sharp), m, st.n_compared,
                            st.n_unconverged, len(st.dcl), _mean(st.dcl), _max(st.dcl),
                            _mean(st.dcd_rel), _max(st.dcd_rel), _mean(st.dcm), _max(st.dcm)])


#: The committed tables-only baseline document; written only by ``baseline-doc``.
BASELINE_DOC = Path("docs/baseline_v1.3.1.md")


def write_report(output_root: Path = Path("benchmark_output"), bench: BenchmarkParams = BENCHMARK,
                 docs_path: Path | None = None) -> list[CaseResult]:
    """Compare all cases and write plots, CSV tables and ``report.md`` under ``output_root``.

    The committed tables-only document is written only when ``docs_path`` is given (the
    ``baseline-doc`` command); ``report`` and ``all`` never touch it. Returns the case results.
    """
    cases = []
    for sec in bench.sections:
        for re in bench.reynolds_low + bench.reynolds_validated:
            try:
                cases.append(compare_case(sec, re, output_root, bench))
            except FileNotFoundError:
                print(f"skipped {sec} Re {re:g}: no reference file")
    cases.sort(key=lambda c: (bench.sections.index(c.section), c.reynolds))
    for c in cases:
        plot_case(c, output_root, bench)
    _tables_csv(cases, output_root / "tables" / "case_metrics.csv")
    (output_root / "report.md").write_text(build_markdown(cases, bench, plots=True))
    if docs_path is not None:
        docs_path.parent.mkdir(parents=True, exist_ok=True)
        docs_path.write_text(build_markdown(cases, bench, plots=False))
    return cases
