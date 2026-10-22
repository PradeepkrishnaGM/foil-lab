"""What does XFOIL's hinge moment contain? (M4 step 3; a diagnostic, not a test)

For flapped sections made by ``Airfoil.with_flap`` this runs XFOIL (as an external program, with
documented commands only: ``SAVE``, ``FNEW``, ``FMOM``, ``CPWR``, ``DUMP``, ``PACC``) and compares
the hinge moment it prints (``FMOM``, the same value as the ``Chinge`` polar column) with
integrals that we compute ourselves from XFOIL's own surface pressure (``CPWR``) and wall shear
(``DUMP``) on XFOIL's own panel nodes (``SAVE``):

* the quadrature of XFOIL's ``CM``, found by trying candidate rules against the polar ``CM``
  (the user guide gives the integral, not the rule);
* the pressure moment about the hinge on the panels aft of the hinge, with that quadrature;
* the friction moment from the ``Cf`` of ``DUMP``, with the convention of ``Cf`` found by
  comparing its integral with ``CD - CDp`` of the polar.

Run: ``python tests/diagnostics/xfoil_hinge_calibration.py [output.md]``. Needs ``xfoil`` and
``xvfb-run``. All numbers are in XFOIL's sign (positive = trailing edge up, counter-clockwise).
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import foil_lab as fl
from foil_lab.benchmark.xfoil import write_coordinate_file

XH, YH_FRAC = 0.75, 0.5


def run_case(coords: np.ndarray, hinge: np.ndarray, alpha: float, reynolds: float, wd: Path) -> dict:
    write_coordinate_file(wd / "a.dat", coords, "a")
    visc = f"VISC {reynolds:g}\nVPAR\nN 9\nXTR 1 1\n\n" if reynolds else ""
    script = (f"LOAD a.dat\nPPAR\nN 160\n\n\nPANE\nSAVE pan.dat\n\n\nOPER\n{visc}ITER 200\n"
              f"FNEW {hinge[0]:.8g} {hinge[1]:.8g}\nFMOM\nPACC\na.pol\n\nALFA {alpha:g}\nFMOM\n"
              f"CPWR cp.txt\nDUMP dump.txt\nPACC\n\nQUIT\n")
    for f in ("pan.dat", "cp.txt", "dump.txt", "a.pol"):
        (wd / f).unlink(missing_ok=True)
    out = subprocess.run(["xvfb-run", "-a", "xfoil"], input=script, capture_output=True, text=True,
                         cwd=wd, timeout=300).stdout
    vals = re.findall(r"Hinge moment/span =\s*(\S+)", out)
    fx = re.findall(r"x-Force\s+/span =\s*(\S+)", out)
    fy = re.findall(r"y-Force\s+/span =\s*(\S+)", out)
    res = {"fmom": float(vals[-1]), "fx": float(fx[-1]), "fy": float(fy[-1])}
    pol = (wd / "a.pol").read_text().splitlines()
    row = [l for l in pol if re.match(r"\s*-?\d+\.\d+\s", l)][-1].split()
    res.update(cl=float(row[1]), cd=float(row[2]), cdp=float(row[3]), cm=float(row[4]),
               chinge=float(row[5]))
    res["nodes"] = np.loadtxt(wd / "pan.dat", skiprows=1)
    res["cp"] = np.loadtxt(wd / "cp.txt", skiprows=1)
    if reynolds:
        res["dump"] = np.loadtxt(wd / "dump.txt", skiprows=1, usecols=range(7), max_rows=len(res["nodes"]))
    return res


def _panel_rules(xy: np.ndarray, cp: np.ndarray):
    """Per-panel force (fx, fy) / q and moment arm for two quadratures of the nodal Cp:
    'mid' (mean Cp times the panel vector) and 'lin' (Cp linear along the panel, exact integral
    of Cp times the linear position)."""
    p0, p1 = xy[:-1], xy[1:]
    d = p1 - p0
    nrm = np.column_stack([d[:, 1], -d[:, 0]])          # outward for counter-clockwise nodes
    c0, c1 = cp[:-1], cp[1:]
    return p0, p1, nrm, c0, c1


def pressure_moment(xy, cp, hinge, rule: str, select: np.ndarray, ref=None) -> float:
    """Counter-clockwise moment about ``hinge`` of the pressure force on the selected panels."""
    p0, p1, nrm, c0, c1 = _panel_rules(xy, cp)
    r0, r1 = p0 - hinge, p1 - hinge
    if rule == "mid":
        f = -(0.5 * (c0 + c1))[:, None] * nrm
        m = 0.5 * ((r0 + r1)[:, 0] * f[:, 1] - (r0 + r1)[:, 1] * f[:, 0])
    elif rule == "lin":
        # integral over s in [0, 1] of Cp(s) r(s) x n ds, Cp and r linear
        w00, w01, w11 = 1 / 3, 1 / 6, 1 / 3
        cr = lambda a, b: a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        m = -(c0 * (w00 * cr(r0, nrm) + w01 * cr(r1, nrm)) + c1 * (w01 * cr(r0, nrm) + w11 * cr(r1, nrm)))
    else:
        raise ValueError(rule)
    return float(np.sum(m[select]))


def pressure_moment_x_arm_only(xy, cp, hinge, select) -> float:
    """Counter-clockwise moment keeping only the chordwise arm: sum (x - x_h) f_y of the pressure
    force on the selected panels (the other term, -(y - y_h) f_x, is dropped)."""
    p0, p1, nrm, c0, c1 = _panel_rules(xy, cp)
    f = -(0.5 * (c0 + c1))[:, None] * nrm
    mid = 0.5 * (p0 + p1)
    return float(np.sum(((mid[:, 0] - hinge[0]) * f[:, 1])[select]))


def force(xy, cp, rule: str, select) -> np.ndarray:
    p0, p1, nrm, c0, c1 = _panel_rules(xy, cp)
    cm_ = 0.5 * (c0 + c1) if rule == "mid" else 0.5 * (c0 + c1)
    return (-cm_[:, None] * nrm)[select].sum(axis=0)


def friction(dump: np.ndarray, hinge, scale: str, select_x: float, alpha_deg: float):
    """Wall-shear force of the surface rows of ``DUMP`` (rows 0..n: the nodes of ``SAVE``).

    ``tau = Cf Ue^2`` (``scale`` 'cf_ue2') or ``Cf`` ('cf'), acting along the flow, i.e. towards
    the trailing edge on both surfaces. XFOIL's ``Ue`` is positive on the first (upper) surface
    and negative on the second in the direction of increasing ``s``, so the flow runs along ``-s``
    where ``Ue > 0``. Returns (counter-clockwise moment, chordwise arm only, of the panels with
    midpoint x > ``select_x``, friction drag ``F . (cos a, sin a)`` of all panels).
    """
    s, x, y, ue, cf = dump[:, 0], dump[:, 1], dump[:, 2], dump[:, 3], dump[:, 6]
    tau = cf * ue ** 2 if scale == "cf_ue2" else cf
    d = np.column_stack([np.diff(x), np.diff(y)])
    t = 0.5 * (tau[1:] + tau[:-1])
    sgn = -np.sign(0.5 * (ue[1:] + ue[:-1]))
    f = (sgn * t)[:, None] * d
    mid = np.column_stack([0.5 * (x[1:] + x[:-1]), 0.5 * (y[1:] + y[:-1])])
    sel = mid[:, 0] > select_x
    m_x = (mid[:, 0] - hinge[0]) * f[:, 1]                    # chordwise arm only
    a = np.radians(alpha_deg)
    return float(np.sum(m_x[sel])), float(np.sum(f @ np.array([np.cos(a), np.sin(a)])))


def main(out_path: str | None = None) -> None:
    lines = ["| section | flap deg | alpha | Re | FMOM | pressure only (mid) | pressure, x arm only | "
             "friction (Cf*Ue^2) | x arm only + friction | CM check (polar / mid / lin) | "
             "CD-CDp check (polar / Cf Ue^2 / Cf) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    with tempfile.TemporaryDirectory(prefix="hinge-cal-") as tmp:
        wd = Path(tmp)
        for name in ("0012", "2412"):
            base = fl.Airfoil.naca(name, 160)
            for d in (0.0, 10.0, 20.0):
                fa = base.with_flap(XH, d, YH_FRAC)
                for alpha, re_ in ((3.0, 0.0), (3.0, 1e6), (6.0, 1e6)):
                    r = run_case(fa.coords, fa.hinge, alpha, re_, wd)
                    xy, cp = r["nodes"], r["cp"][:, 1]
                    n = len(xy) - 1
                    allp = np.ones(n, bool)
                    mid_x = 0.5 * (xy[:-1, 0] + xy[1:, 0])
                    aft = mid_x > XH
                    # CM about (0.25, 0) with the two quadratures (XFOIL's CM is nose-up positive)
                    a = np.radians(alpha)
                    cm_mid = -pressure_moment(xy, cp, np.array([0.25, 0.0]), "mid", allp)
                    cm_lin = -pressure_moment(xy, cp, np.array([0.25, 0.0]), "lin", allp)
                    p_mid = pressure_moment(xy, cp, fa.hinge, "mid", aft)
                    p_lin = pressure_moment(xy, cp, fa.hinge, "lin", aft)
                    p_x = pressure_moment_x_arm_only(xy, cp, fa.hinge, aft)
                    if re_:
                        dmp = r["dump"][:len(xy)]            # surface rows; the wake rows follow
                        m_f, cdf_u = friction(dmp, fa.hinge, "cf_ue2", XH, alpha)
                        _, cdf_c = friction(dmp, fa.hinge, "cf", XH, alpha)
                        cdf_txt = f"{r['cd'] - r['cdp']:.5f} / {cdf_u:.5f} / {cdf_c:.5f}"
                    else:
                        m_f, cdf_txt = 0.0, "-"
                    lines.append(
                        f"| NACA {name} | {d:g} | {alpha:g} | {re_:g} | {r['fmom']:.6f} | {p_mid:.6f} | "
                        f"{p_x:.6f} | {m_f:.6f} | {p_x + m_f:.6f} | "
                        f"{r['cm']:.4f} / {cm_mid:.4f} / {cm_lin:.4f} | {cdf_txt} |")
                    print(lines[-1], flush=True)
    text = "\n".join(lines) + "\n"
    if out_path:
        Path(out_path).write_text(text)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
