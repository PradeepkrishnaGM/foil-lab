"""Flap cases of the benchmark (SPEC.md 4.1 and 4.5; DECISIONS.md, M4).

A flap case is a canonical section with a plain flap (``<section>_F<deflection>``, for example
``NACA2412_F10``; :func:`foil_lab.benchmark.sections.flap_name`), made by ``Airfoil.with_flap`` and
written once to a coordinate file that both codes read. XFOIL gets the hinge point with ``FNEW`` and
reports its hinge moment (``Chinge``, trailing edge up positive); Foil-Lab's ``ch`` is trailing edge
down positive.

Two hinge-moment comparisons (DECISIONS.md, M4 decisions after the pilot):

* (a) the solvers without the definition question: XFOIL's own surface pressure (``CPWR``) and
  wall shear (``DUMP``, ``Cf Ue^2`` along the flow) at every converged point, integrated with
  Foil-Lab's definition (the flap panels, about the hinge point, pressure plus friction);
* (b) Foil-Lab's ``ch`` against ``-Chinge`` as XFOIL reports it, whose definition is not known
  (docs/xfoil_hinge_calibration.md).

``python -m foil_lab.benchmark.flap_run pilot|full`` runs XFOIL (polars at 160 and 240 PANE nodes,
and the surface dumps) and Foil-Lab on the cases and writes the tables.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from ..params import BENCHMARK, XFOIL, BenchmarkParams
from .references import _one, reference_name
from .report import MethodStats, attached_filter
from .sections import canonical, flap_name
from .viscous_run import (CLOSURE_SETS, TARGETS, _write, method_name, read_case, run_case,
                          unreliable_quantities, verdict)
from .xfoil import XfoilDumps, read_reference_csv, run_dumps, write_coordinate_file

#: loose target on the comparison (b) against -Chinge (user, after the pilot)
TARGET_B = {"mean": 0.0025, "max": 0.004}


def case_names(bench: BenchmarkParams = BENCHMARK) -> list[str]:
    """All flap cases of the matrix."""
    return [flap_name(b, d) for b in bench.flap_bases for d in bench.flap_deflections]


def _ref_path(ref_dir: Path, name: str, re: float, nodes: int) -> Path:
    return ref_dir / reference_name(name, re, nodes)


def _dump_path(out_dir: Path, name: str, re: float, nodes: int) -> Path:
    return out_dir / "xfoil_dumps" / reference_name(name, re, nodes).replace(".csv", ".npz")


def _ours_path(out_dir: Path, name: str, re: float) -> Path:
    return out_dir / "ours" / f"{name}_Re{int(round(re))}.csv"


# --------------------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------------------- #
def _dump_one(sec, re: float, nodes: int, out_dir: str) -> tuple[str, str, float]:
    """Worker: XFOIL surface dumps of one case, saved compactly."""
    t0 = time.perf_counter()
    params = dataclasses.replace(XFOIL, pane_nodes=nodes)
    path = _dump_path(Path(out_dir), sec.name, re, nodes)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        d = run_dumps(sec.coords, sec.name, re, sec.hinge, params=params)
    except Exception as exc:                                 # recorded, the case then has no (a)
        return path.name, f"failed: {exc}", time.perf_counter() - t0
    d.to_npz(path)
    return path.name, f"{len(d.alphas)} points", time.perf_counter() - t0


def run_references(cases: list[tuple[str, float]], ref_dir: Path, out_dir: Path,
                   bench: BenchmarkParams = BENCHMARK, workers: int | None = None) -> None:
    """XFOIL polars (160 and 240 PANE nodes) and surface dumps of the cases ``(name, Re)``."""
    ref_dir.mkdir(parents=True, exist_ok=True)
    secs = {}
    for name, _ in cases:
        if name not in secs:
            secs[name] = canonical(name, bench.num_panels)
            (out_dir / "coords").mkdir(parents=True, exist_ok=True)
            write_coordinate_file(out_dir / "coords" / f"{name}.dat", secs[name].coords, name)
    with ProcessPoolExecutor(max_workers=workers or bench.workers) as pool:
        futs = [pool.submit(_one, secs[n], re, nodes, str(ref_dir), str(out_dir / "xfoil_logs"))
                for n, re in cases for nodes in bench.pane_nodes]
        futs += [pool.submit(_dump_one, secs[n], re, nodes, str(out_dir))
                 for n, re in cases for nodes in bench.pane_nodes]
        for i, fut in enumerate(as_completed(futs), 1):
            name, status, secs_ = fut.result()
            print(f"[xfoil {i}/{len(futs)}] {name}: {status}, {secs_:.0f} s", flush=True)


def run_ours(cases: list[tuple[str, float]], out_dir: Path, workers: int | None = None,
             bench: BenchmarkParams = BENCHMARK) -> None:
    """Foil-Lab sweeps of the cases (one CSV per case, with the hinge moment)."""
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    ctx = mp.get_context("spawn")
    todo = [(n, re) for n, re in cases if not _ours_path(out_dir, n, re).exists()]
    with ProcessPoolExecutor(max_workers=workers or bench.workers, mp_context=ctx) as pool:
        futs = [pool.submit(run_case, n, re, CLOSURE_SETS[0]) for n, re in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            name, re, rows = fut.result()
            _write(_ours_path(out_dir, name, re), rows)
            print(f"[ours {i}/{len(futs)}] {name} Re {re:g}: "
                  f"{sum(r['converged'] == '1' for r in rows)}/{len(rows)} converged, "
                  f"{sum(r['runtime_s'] for r in rows):.0f} s", flush=True)


# --------------------------------------------------------------------------------------- #
# Hinge moment from XFOIL's own surface data, with Foil-Lab's definition
# --------------------------------------------------------------------------------------- #
def xfoil_hinge_moment(d: XfoilDumps, coords: np.ndarray, flap_panels: np.ndarray,
                       hinge: np.ndarray) -> dict[float, tuple[float, float]]:
    """Comparison (a): ``{alpha: (ch pressure, ch friction)}`` from XFOIL's ``Cp`` and ``Cf`` at its
    nodes with Foil-Lab's definition: the panels of XFOIL's node set that lie on the flap of
    ``coords`` (each panel takes the flag of the nearest segment of ``coords``), moment about the
    hinge point of the pressure force (mean ``Cp`` of the two nodes times normal times length) and of
    the wall shear (``tau = Cf Ue^2`` along the flow, towards the trailing edge on both surfaces; XFOIL's
    ``Ue`` is positive on the first surface and negative on the second), divided by q and the unit
    chord squared; positive trailing edge down (so the sign of XFOIL's counter-clockwise moment is
    reversed).
    """
    xy = d.nodes
    p0, p1 = xy[:-1], xy[1:]
    dv = p1 - p0
    mid = 0.5 * (p0 + p1)
    # flag of each XFOIL panel: that of the nearest segment of our flapped contour
    a, b = coords[:-1][None], coords[1:][None]
    ab = b - a
    t = np.clip(np.sum((mid[:, None] - a) * ab, axis=2) / np.maximum(np.sum(ab * ab, axis=2), 1e-300), 0, 1)
    dist = np.sum((mid[:, None] - (a + t[..., None] * ab)) ** 2, axis=2)
    sel = np.asarray(flap_panels)[np.argmin(dist, axis=1)]
    nrm = np.column_stack([dv[:, 1], -dv[:, 0]])             # outward for the counter-clockwise nodes
    r = mid - hinge
    out = {}
    for k, alpha in enumerate(d.alphas):
        cp = 0.5 * (d.cp[k][:-1] + d.cp[k][1:])
        f = -cp[:, None] * nrm
        mz_p = np.sum((r[:, 0] * f[:, 1] - r[:, 1] * f[:, 0])[sel])
        ue, cf = d.ue[k], d.cf[k]
        tau = 0.5 * (cf[1:] + cf[:-1]) * (0.5 * (ue[1:] + ue[:-1])) ** 2
        fs = (-np.sign(0.5 * (ue[1:] + ue[:-1])) * tau)[:, None] * dv
        mz_f = np.sum((r[:, 0] * fs[:, 1] - r[:, 1] * fs[:, 0])[sel])
        out[alpha] = (-float(mz_p), -float(mz_f))
    return out


# --------------------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------------------- #
@dataclasses.dataclass
class FlapCase:
    name: str
    base: str
    deflection: float
    re: float
    rows: list                      # per angle (dicts)
    stats: MethodStats              # SPEC 4.1 metrics inside the filter, reliability rule applied
    n_filter: int
    n_xfoil: int                    # filter angles where XFOIL converged
    n_both: int
    cut: float | None
    n_unrel: dict
    d_a: list                       # |ch - ch(XFOIL Cp, Cf; our definition)| at filter points
    d_b: list                       # |ch + Chinge|
    sens_a: list                    # |a(240) - a(160)| where both exist
    sens_b: list                    # |Chinge(240) - Chinge(160)|
    ch_abs: list                    # |ch| of XFOIL's -Chinge at the compared points (scale)
    friction_share: list            # (|friction part|, |ch|) of ours at the compared points
    unconv_angles: list             # filter angles XFOIL converged and Foil-Lab did not
    runtime_total: float
    xfoil_runtime: float
    n_all_ours: int                 # converged angles of all requested
    n_all: int
    n_all_xfoil: int
    a_available: bool


def _stats(v) -> tuple[float, float]:
    v = [x for x in v if math.isfinite(x)]
    return (sum(v) / len(v), max(v)) if v else (math.nan, math.nan)


def compare_case(name: str, re: float, ref_dir: Path, out_dir: Path,
                 bench: BenchmarkParams = BENCHMARK) -> FlapCase:
    """Everything the report needs for one case."""
    ref = read_reference_csv(_ref_path(ref_dir, name, re, bench.pane_nodes[0]).read_text())
    p240 = _ref_path(ref_dir, name, re, bench.pane_nodes[1])
    ref240 = read_reference_csv(p240.read_text()) if p240.exists() else None
    ours = {r["alpha"]: r for r in read_case(_ours_path(out_dir, name, re))
            if r["method"] == method_name(CLOSURE_SETS[0])}
    sec = canonical(name, bench.num_panels)
    dumps = {}
    for n in bench.pane_nodes:
        p = _dump_path(out_dir, name, re, n)
        if p.exists():
            d = XfoilDumps.from_npz(p)
            dumps[n] = xfoil_hinge_moment(d, sec.coords, sec.flap_panels, np.asarray(sec.hinge))
    filt, cut = attached_filter(ref, bench)
    unrel = {a: unreliable_quantities(ref.rows[a], None if ref240 is None else ref240.rows.get(a))
             for a in filt}
    st = MethodStats()
    rows, d_a, d_b, s_a, s_b, ch_abs, fshare, unconv = [], [], [], [], [], [], [], []
    a160 = dumps.get(bench.pane_nodes[0], {})
    a240 = dumps.get(bench.pane_nodes[1], {})
    for a in ref.alphas:
        x = ref.rows[a]
        x2 = ref240.rows.get(a) if ref240 else None
        o = ours.get(a)
        conv = o is not None and o["converged"] == "1" and math.isfinite(o["cl"])
        in_f = a in filt
        ca = sum(a160[a]) if a in a160 else math.nan
        ca240 = sum(a240[a]) if a in a240 else math.nan
        rows.append(dict(alpha=a, in_filter=in_f, xfoil=x is not None, ours=conv,
                         cl_x=x.cl if x else math.nan, cd_x=x.cd if x else math.nan,
                         cm_x=x.cm if x else math.nan, ch_x=x.ch if x else math.nan,
                         ch_xdef=ca, cl=o["cl"] if conv else math.nan, cd=o["cd"] if conv else math.nan,
                         cm=o["cm"] if conv else math.nan, ch=o["ch"] if conv else math.nan,
                         ch_friction=o["ch_friction"] if conv else math.nan,
                         runtime=o["runtime_s"] if o else math.nan))
        if not (in_f and x is not None):
            continue
        st.n_compared += 1
        if o is not None:
            st.runtime.append(o["runtime_s"])
        if not conv:
            st.n_unconverged += 1
            unconv.append(a)
            continue
        skip = unrel[a]
        if "cl" not in skip:
            st.dcl.append(abs(o["cl"] - x.cl))
        if "cm" not in skip:
            st.dcm.append(abs(o["cm"] - x.cm))
        if "cd" not in skip:
            st.dcd_rel.append(abs(o["cd"] - x.cd) / x.cd)
        if x.ch is not None:
            d_b.append(abs(o["ch"] - x.ch))
            ch_abs.append(abs(x.ch))
            fshare.append((abs(o["ch_friction"]), abs(o["ch"])))
            if x2 is not None and x2.ch is not None:
                s_b.append(abs(x2.ch - x.ch))
        if math.isfinite(ca):
            d_a.append(abs(o["ch"] - ca))
            if math.isfinite(ca240):
                s_a.append(abs(ca240 - ca))
    runs = ref.provenance.get("runs", "")
    xr = sum(float(p.split(",")[-1].strip()[:-2]) for p in runs.split(";") if p.strip().endswith(" s"))
    n_unrel = {q: sum(q in unrel[a] for a in filt if ref.rows[a] is not None)
               for q in ("cl", "cd", "cm")}
    base = name.rsplit("_F", 1)[0]
    return FlapCase(
        name, base, float(name.rsplit("_F", 1)[1]), re, rows, st, len(filt),
        sum(ref.rows[a] is not None for a in filt),
        sum(1 for r in rows if r["in_filter"] and r["xfoil"] and r["ours"]), cut, n_unrel, d_a, d_b, s_a,
        s_b, ch_abs, fshare, unconv, sum(r["runtime"] for r in rows if math.isfinite(r["runtime"])), xr,
        sum(1 for r in rows if r["ours"]), len(rows), sum(1 for r in rows if r["xfoil"]), bool(a160))


def _pool(cases: list[FlapCase]) -> MethodStats:
    out = MethodStats()
    for c in cases:
        out.n_compared += c.stats.n_compared
        out.n_unconverged += c.stats.n_unconverged
        for k in ("dcl", "dcm", "dcd_rel", "runtime"):
            getattr(out, k).extend(getattr(c.stats, k))
    return out


def _md(head: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
                     + ["| " + " | ".join(r) + " |" for r in rows]) + "\n"


def _failed(v: dict) -> list[str]:
    return [k for k, (val, ok) in v.items() if not (ok and math.isfinite(val))]


#: leave-one-out cases of the interpolation in deflection: (left-out deflection, deflections used or None = all others)
LOO_CASES = ((10.0, None), (15.0, (10.0, 20.0)))


def loo_errors(cs: list[FlapCase]) -> list[list[str]]:
    """Leave-one-out error of :func:`foil_lab.flap_polars.interpolate_in_deflection` on the Foil-Lab
    points of the benchmark: for each section and Reynolds number the polar of the left-out
    deflection is interpolated from the others and compared with the solved one. Rows: (left out, used,
    Re) with mean / max |dCl|, |dCm|, |d ch| and relative dCd, over the angles where every point involved
    converged, for all angles and for the attached-flow filter angles of the left-out case."""
    from ..flap_polars import interpolate_in_deflection
    rows = []
    res = sorted({c.re for c in cs})
    for tgt, use in LOO_CASES:
        for r in res:
            errs = {"all": {q: [] for q in ("cl", "cd", "cm", "ch")}, "filter": {q: [] for q in ("cl", "cd", "cm", "ch")}}
            for base in sorted({c.base for c in cs}):
                grp = {c.deflection: c for c in cs if c.base == base and c.re == r}
                if tgt not in grp or len(grp) < 3:
                    continue
                used = sorted(d for d in grp if d != tgt and (use is None or d in use))
                if len(used) < 2:
                    continue
                for q in ("cl", "cd", "cm", "ch"):
                    tab = np.array([[row[q] for row in grp[d].rows] for d in used])
                    ref = np.array([row[q] for row in grp[tgt].rows])
                    est = interpolate_in_deflection(np.array(used), tab, tgt)
                    flt = np.array([row["in_filter"] for row in grp[tgt].rows])
                    for key, mask in (("all", np.ones_like(flt)), ("filter", flt)):
                        m = np.isfinite(est) & np.isfinite(ref) & mask
                        e = np.abs(est - ref)[m] / (np.abs(ref[m]) if q == "cd" else 1.0)
                        errs[key][q].extend(e.tolist())
            f = lambda v, fmt="{:.4f}": (fmt.format(np.mean(v)) + " / " + fmt.format(np.max(v))) if v else "-"
            rows.append([f"{tgt:g}", "all others" if use is None else "/".join(f"{u:g}" for u in use), f"{r:g}",
                         str(len(errs["all"]["cl"])), f(errs["all"]["cl"]), f(errs["all"]["cm"]),
                         f(errs["all"]["ch"], "{:.5f}"), f([x * 100 for x in errs["all"]["cd"]], "{:.1f}") + " %",
                         f(errs["filter"]["cl"]), f(errs["filter"]["cm"]), f(errs["filter"]["ch"], "{:.5f}"),
                         f([x * 100 for x in errs["filter"]["cd"]], "{:.1f}") + " %"])
    return rows


def build_report(cases: list[tuple[str, float]], ref_dir: Path, out_dir: Path,
                 bench: BenchmarkParams = BENCHMARK, detail: bool = False) -> str:
    """The M4 flap report as markdown."""
    cs = [compare_case(n, re, ref_dir, out_dir, bench) for n, re in cases]
    defl = sorted({c.deflection for c in cs})
    res = sorted({c.re for c in cs})
    lines = ["# M4 flap benchmark\n",
             f"Cases: {len(cs)} ({', '.join(sorted({c.base for c in cs}))}; deflections "
             f"{', '.join(f'{d:g}' for d in defl)} deg; Re {', '.join(f'{r:g}' for r in res)}). 25 angles each, "
             f"-4 to +20 deg. Hinge {bench.flap_hinge_x:g} c at {bench.flap_hinge_y:g} of the thickness.\n"]

    # --- 1. SPEC 4.1 pass / fail per deflection and Re (sections pooled, as in the main benchmark)
    verdicts = {}
    rows = []
    for d in defl:
        row = [f"{d:g}"]
        for r in res:
            sel = [c for c in cs if c.deflection == d and c.re == r]
            v = verdict(_pool(sel))
            verdicts[(d, r)] = v
            bad = _failed(v)
            row.append("✓" if not bad else "✗ " + ", ".join(bad))
        rows.append(row)
    lines += ["## 1. SPEC.md 4.1 targets per deflection and Reynolds number\n",
              "Cl, Cd, Cm; attached-flow filter, reference-reliability rule, both sections pooled at each "
              "(deflection, Re) as in the main benchmark. Targets: converged >= 95 %, mean / max dCl "
              f"{TARGETS['cl_mean']} / {TARGETS['cl_max']}, mean / max rel dCd {TARGETS['cd_mean']:.0%} / "
              f"{TARGETS['cd_max']:.0%}, mean / max dCm {TARGETS['cm_mean']} / {TARGETS['cm_max']}.\n",
              _md(["flap deg"] + [f"Re {r:g}" for r in res], rows)]
    ok = {d: all(not _failed(verdicts[(d, r)]) for r in res) for d in defl}
    run_ok = None
    for d in defl:
        if ok[d]:
            run_ok = d
        else:
            break
    top = max((d for d in defl if ok[d]), default=None)
    lines += [f"**Validated flap range** (largest deflection at which every Re meets every target, all "
              f"smaller deflections too): {'none' if run_ok is None else f'{run_ok:g} deg'}. "
              f"Largest single deflection that passes: {'none' if top is None else f'{top:g} deg'}. "
              "Deflections beyond the validated range are benchmarked without a claim.\n"]

    # --- 2. numbers behind it
    rows = []
    for d in defl:
        for r in res:
            st = _pool([c for c in cs if c.deflection == d and c.re == r])
            v = verdicts[(d, r)]
            f = lambda k, fmt, scale=1.0: f"{v[k][0] * scale:{fmt}}" + ("" if v[k][1] else " ✗")
            rows.append([f"{d:g}", f"{r:g}", str(st.n_compared), f"{v['converged'][0]:.1%}" + ("" if v['converged'][1] else " ✗"),
                         f"{f('mean dCl', '.4f')} / {f('max dCl', '.4f')}",
                         f"{f('mean rel dCd', '.1f', 100)} % / {f('max rel dCd', '.1f', 100)} %",
                         f"{f('mean dCm', '.4f')} / {f('max dCm', '.4f')}"])
    lines += ["### Numbers\n", _md(["flap deg", "Re", "filter points (XFOIL conv.)", "Foil-Lab converged",
                                    "mean / max dCl", "mean / max rel dCd", "mean / max dCm"], rows)]

    # --- 3. convergence per case
    rows = [[c.name, f"{c.re:g}", f"{c.n_all_ours}/{c.n_all}", f"{c.n_all_xfoil}/{c.n_all}",
             f"{c.n_both}/{c.n_xfoil} (cut {c.cut})", f"{c.stats.n_unconverged}" + (f" ({', '.join(f'{a:g}' for a in c.unconv_angles)})" if c.unconv_angles else ""),
             f"{c.n_unrel['cl']}/{c.n_unrel['cd']}/{c.n_unrel['cm']}"] for c in cs]
    lines += ["## 2. Convergence per case\n", "Foil-Lab and XFOIL converged out of all 25 angles; both inside the "
              "filter (XFOIL converged there); Foil-Lab unconverged at filter points XFOIL converged; reference "
              "points not counted by the reliability rule (Cl/Cd/Cm).\n",
              _md(["case", "Re", "Foil-Lab all", "XFOIL all", "both in filter", "ours unconv. in filter", "unreliable"], rows)]

    # --- 4. hinge moment
    rows = []
    for d in defl:
        for r in res:
            sel = [c for c in cs if c.deflection == d and c.re == r]
            ma, xa = _stats([x for c in sel for x in c.d_a])
            mb, xb = _stats([x for c in sel for x in c.d_b])
            sa = _stats([x for c in sel for x in c.sens_a])
            sb = _stats([x for c in sel for x in c.sens_b])
            sc = _stats([x for c in sel for x in c.ch_abs])[0]
            okb = mb <= TARGET_B["mean"] and xb <= TARGET_B["max"]
            rows.append([f"{d:g}", f"{r:g}", f"{ma:.5f} / {xa:.5f}", f"{sa[0]:.5f} / {sa[1]:.5f}",
                         f"{mb:.5f} / {xb:.5f}" + ("" if okb else " ✗"), f"{sb[0]:.5f} / {sb[1]:.5f}", f"{sc:.4f}"])
    pa = [x for c in cs for x in c.d_a]
    pb = [x for c in cs for x in c.d_b]
    lines += ["## 3. Hinge moment\n",
              "(a) Foil-Lab `ch` against XFOIL's Cp and Cf integrated with Foil-Lab's definition; (b) Foil-Lab "
              f"`ch` against `-Chinge` of XFOIL, loose target mean <= {TARGET_B['mean']}, max <= {TARGET_B['max']} "
              "(definition of Chinge unexplained, docs/xfoil_hinge_calibration.md). Mean / max absolute "
              "difference over the filter points where both converged, both sections pooled; XFOIL's own "
              "change between 240 and 160 PANE nodes next to each; last column the mean |ch| (XFOIL) as scale.\n",
              _md(["flap deg", "Re", "(a) mean / max", "(a) XFOIL 240 vs 160", "(b) mean / max", "(b) XFOIL 240 vs 160",
                   "mean abs ch"], rows),
              f"All cases pooled: (a) mean {_stats(pa)[0]:.5f}, max {_stats(pa)[1]:.5f} over {len(pa)} points; "
              f"(b) mean {_stats(pb)[0]:.5f}, max {_stats(pb)[1]:.5f} over {len(pb)} points. "
              f"Friction share of Foil-Lab's ch (mean |friction part| over mean |ch|, compared points): "
              f"{sum(x[0] for c in cs for x in c.friction_share) / max(sum(x[1] for c in cs for x in c.friction_share), 1e-300):.2%}.\n"]
    # quantiles of (a)
    if pa:
        q = np.quantile(pa, [0.5, 0.9, 0.95, 0.99, 1.0])
        sa_all = [x for c in cs for x in c.sens_a]
        lines += [f"(a) quantiles 50/90/95/99/100 %: {', '.join(f'{x:.5f}' for x in q)}. XFOIL 240 vs 160 of the (a) "
                  f"integral: mean {_stats(sa_all)[0]:.5f}, 95 % {np.quantile(sa_all, 0.95):.5f}, max {_stats(sa_all)[1]:.5f}.\n"]

    # --- 5. run times
    rows = [[f"{d:g}", f"{r:g}", f"{np.mean([c.runtime_total for c in cs if c.deflection == d and c.re == r]):.0f}",
             f"{np.mean([c.xfoil_runtime for c in cs if c.deflection == d and c.re == r]):.0f}"] for d in defl for r in res]
    tot = sum(c.runtime_total for c in cs)
    npts = sum(c.n_all for c in cs)
    lines += ["## 4. Run time\n", f"Foil-Lab: {tot:.0f} s of solver time in total over {npts} points "
              f"({tot / max(npts, 1):.1f} s per point, several workers at once). Mean per case (s):\n",
              _md(["flap deg", "Re", "Foil-Lab (25 angles)", "XFOIL polar processes"], rows)]
    lines += ["## 5. Interpolation in deflection (leave-one-out)\n",
              "The polar of the left-out deflection interpolated (PCHIP in the deflection, per angle) from the "
              "others and compared with the solved polar; Foil-Lab points, both sections pooled; mean / max "
              "absolute error (Cd: relative), over all angles where every point involved converged, and over "
              "the attached-flow filter angles.\n",
              _md(["left out", "from", "Re", "points", "dCl", "dCm", "d ch", "rel dCd", "dCl (filter)",
                   "dCm (filter)", "d ch (filter)", "rel dCd (filter)"], loo_errors(cs))]
    if detail:
        for c in cs:
            lines += [f"\n#### {c.name}, Re {c.re:g}\n",
                      _md(["alpha", "filter", "Cl X / ours", "Cd X / ours", "Cm X / ours", "ch: -Chinge / XFOIL Cp,Cf / ours"],
                          [[f"{r['alpha']:g}", "y" if r["in_filter"] else "", f"{r['cl_x']:.4f} / {r['cl']:.4f}",
                            f"{r['cd_x']:.5f} / {r['cd']:.5f}", f"{r['cm_x']:.4f} / {r['cm']:.4f}",
                            f"{r['ch_x']:.5f} / {r['ch_xdef']:.5f} / {r['ch']:.5f}"] for r in c.rows])]
    return "\n".join(str(x) for x in lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=("pilot", "full", "report"))
    ap.add_argument("--out", type=Path, default=Path("benchmark_output/m4"))
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--skip-xfoil", action="store_true")
    ap.add_argument("--skip-ours", action="store_true")
    ap.add_argument("--detail", action="store_true", help="per-angle tables in the report")
    args = ap.parse_args(argv)
    bench = BENCHMARK
    if args.what == "pilot":
        cases = [(flap_name("NACA2412", d), re) for d in (10.0, 20.0) for re in (1e6, 3e6)]
    else:
        cases = [(n, re) for n in case_names(bench) for re in bench.reynolds_validated]
    ref_dir = args.out / "xfoil_ref"
    t0 = time.perf_counter()
    if args.what != "report":
        if not args.skip_xfoil:
            run_references(cases, ref_dir, args.out, bench, args.workers)
        if not args.skip_ours:
            run_ours(cases, args.out, args.workers, bench)
    text = build_report(cases, ref_dir, args.out, bench, args.detail)
    (args.out / f"{args.what}.md").write_text(text)
    print(text)
    print(f"total {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
