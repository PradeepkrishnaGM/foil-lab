"""Plain flap geometry (SPEC.md 4.5; DECISIONS.md, M4).

A plain flap is the part of a section aft of the vertical cut at the hinge position ``hinge_x``,
rotated rigidly about the hinge point H by the deflection (trailing edge down positive). The hinge
point is at the fraction ``hinge_y`` of the local thickness (0 lower surface, 1 upper surface).

The contour that results has no gap and a continuous tangent:

* On the side that opens (upper surface for a trailing-edge-down deflection) the main surface and
  the rotated flap surface both end on the circle about H through the surface point of the cut, so
  a circular arc about H joins them. Short cubic fairings (Bezier curves with the end tangents of
  the arc and of the surface) join the arc to the surfaces, so the tangent is continuous. The
  circle is the kinematic seal: pressure on it points through H and has no moment about H.
* On the side that closes the rotated flap surface runs into the main body. The two surface curves
  are intersected, the part inside the body is dropped, and the corner is rounded by a fillet (a
  cubic Bezier with the end tangents of the two surfaces, size ``flap_fillet_radius``; 0 keeps
  the plain notch).
* When the hinge point lies closer than ``flap_min_arc_radius`` to a surface it is moved onto that
  surface; there is no arc on that side then, and the corner is rounded by the same fillet.

Negative deflections use the same construction on the mirror image of the section.

The nodes of a flapped section come from this construction, not from resampling: pieces of the
contour are sampled by arc length with the cosine spacing of :func:`foil_lab.geometry.repanel_cosine`
plus a refinement around the hinge zones (``flap_zone_panels`` panels across each zone, spacing
growing by at most ``flap_grading`` per panel). Every junction of two pieces is a node, so each
panel belongs to exactly one zone: main part, flap, or hinge arc.

The section keeps the reference frame of the undeflected section (leading edge, chord, quarter
chord), because the flapped section stores the undeflected one as ``FlapInfo.base``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.optimize import brentq, root

from . import geometry as geo
from .params import GEOMETRY, GeometryParams

ZONE_MAIN, ZONE_FLAP, ZONE_ARC = 0, 1, 2


@dataclass(frozen=True)
class FlapInfo:
    """How a flapped ``Airfoil`` was made (everything needed to rebuild it).

    ``base`` is the undeflected section in the current frame (the flapped section's coordinates
    live in the same frame, with the leading edge and chord direction of ``base``). ``zones`` has
    one entry per panel: 0 main part, 1 flap, 2 hinge arc (the arc moves with the flap). ``hinge``
    is the hinge point after any move onto a surface (see the module docstring).
    """

    base: np.ndarray
    hinge_x: float
    hinge_y: float
    deflection: float
    hinge: np.ndarray
    zones: np.ndarray
    params: GeometryParams
    diagnostics: dict = field(default_factory=dict)

    @property
    def flap_panels(self) -> np.ndarray:
        """Boolean mask of the panels that belong to the flap (flap part and hinge arc)."""
        return self.zones >= ZONE_FLAP

    def transformed(self, fn: Callable[[np.ndarray], np.ndarray]) -> "FlapInfo":
        """Same flap after a point transformation ``fn`` (scale, translate, rotate) of the section."""
        base = np.array(fn(self.base))
        hinge = np.array(fn(self.hinge[None, :])[0])
        base.setflags(write=False)
        hinge.setflags(write=False)
        return FlapInfo(base, self.hinge_x, self.hinge_y, self.deflection, hinge, self.zones,
                        self.params, self.diagnostics)


# --------------------------------------------------------------------------------------- #
# Section in its chord frame
# --------------------------------------------------------------------------------------- #
class _Local:
    """The arc-length spline of a section in its chord frame (leading edge at the origin, unit
    chord along x), optionally mirrored about the chord line (then the lower surface is the
    upper one and the arc-length parameter runs the other way)."""

    def __init__(self, coords: np.ndarray, mirror: bool = False):
        self.sp = geo._ArcSpline(coords)
        le, e, chord = self.sp.chord_frame()
        self.le, self.e, self.chord = le, e, chord
        self.nrm = np.array([-e[1], e[0]])
        self.frame = (le, e, self.nrm, chord)
        self.mirror = mirror
        self.length = self.sp.length
        self.s_le = self.length - self.sp.s_le if mirror else self.sp.s_le

    def _raw(self, s):
        d = self.sp.point(s) - self.le
        return np.column_stack([d @ self.e, d @ self.nrm]) / self.chord

    def point(self, s) -> np.ndarray:
        s = np.atleast_1d(np.asarray(s, dtype=float))
        if self.mirror:
            p = self._raw(self.length - s)
            p[:, 1] = -p[:, 1]
            return p
        return self._raw(s)

    def tangent(self, s) -> np.ndarray:
        """Unit tangent in the direction of increasing ``s``."""
        s = np.atleast_1d(np.asarray(s, dtype=float))
        ss = self.length - s if self.mirror else s
        d = np.column_stack([self.sp.x(ss, 1), self.sp.y(ss, 1)])
        t = np.column_stack([d @ self.e, d @ self.nrm])
        t /= np.hypot(t[:, 0], t[:, 1])[:, None]
        if self.mirror:
            t = -t
            t[:, 1] = -t[:, 1]
        return t

    def s_at_x(self, x: float, upper: bool) -> float:
        a, b = (0.0, self.s_le) if upper else (self.s_le, self.length)
        return float(brentq(lambda s: float(self.point(s)[0, 0]) - x, a, b,
                            xtol=1e-15 * self.length, rtol=8.9e-16))


def _rot(p: np.ndarray, h: np.ndarray, phi: float) -> np.ndarray:
    c, s = np.cos(phi), np.sin(phi)
    d = np.atleast_2d(p) - h
    return h + np.column_stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]])


def _rotv(v: np.ndarray, phi: float) -> np.ndarray:
    c, s = np.cos(phi), np.sin(phi)
    v = np.atleast_2d(v)
    return np.column_stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]])


# --------------------------------------------------------------------------------------- #
# Pieces of the contour
# --------------------------------------------------------------------------------------- #
@dataclass
class _Piece:
    f: Callable[[np.ndarray], np.ndarray]     # t in [0, 1] -> points
    zone: int
    kind: str                                 # "surface", "arc", "fairing", "fillet"


_M = 1200                                     # cells per piece in the length tables


def _tables(piece: _Piece):
    """Parameter and arc-length tables of a piece on a grid clustered at both ends."""
    th = np.linspace(0.0, 1.0, _M + 1)
    t = 0.5 * (1.0 - np.cos(np.pi * th))
    p = piece.f(t)
    l = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))))
    return t, l


def _surface_piece(L: _Local, s0: float, s1: float, zone: int, rot=None) -> _Piece:
    def f(t):
        p = L.point(s0 + np.asarray(t) * (s1 - s0))
        return p if rot is None else _rot(p, *rot)
    return _Piece(f, zone, "surface")


def _bezier(p0, t0, p1, t1, zone, kind) -> _Piece:
    p0, p1 = np.asarray(p0, float).ravel(), np.asarray(p1, float).ravel()
    h = np.hypot(*(p1 - p0)) / 3.0
    c0, c1 = p0 + h * np.asarray(t0, float).ravel(), p1 - h * np.asarray(t1, float).ravel()

    def f(t):
        t = np.asarray(t, float)[:, None]
        return ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * c0 + 3 * (1 - t) * t ** 2 * c1
                + t ** 3 * p1)
    return _Piece(f, zone, kind)


def _split_bezier(piece_args, zones) -> list[_Piece]:
    """Two halves (de Casteljau at t = 1/2) of a cubic Bezier, so a fillet can belong to two zones."""
    p0, t0, p1, t1 = (np.asarray(a, float).ravel() for a in piece_args)
    h = np.hypot(*(p1 - p0)) / 3.0
    q = [p0, p0 + h * t0, p1 - h * t1, p1]
    a = [0.5 * (q[i] + q[i + 1]) for i in range(3)]
    b = [0.5 * (a[i] + a[i + 1]) for i in range(2)]
    m = 0.5 * (b[0] + b[1])
    first, second = [q[0], a[0], b[0], m], [m, b[1], a[2], q[3]]

    def mk(c, zone):
        def f(t):
            t = np.asarray(t, float)[:, None]
            return ((1 - t) ** 3 * c[0] + 3 * (1 - t) ** 2 * t * c[1] + 3 * (1 - t) * t ** 2 * c[2]
                    + t ** 3 * c[3])
        return _Piece(f, zone, "fillet")
    return [mk(first, zones[0]), mk(second, zones[1])]


def _arc_piece(h, r, th0, th1, zone) -> _Piece:
    def f(t):
        th = th0 + np.asarray(t, float) * (th1 - th0)
        return h + r * np.column_stack([np.cos(th), np.sin(th)])
    return _Piece(f, zone, "arc")


def _turn(t0, t1) -> float:
    """Angle between two unit tangents."""
    t0, t1 = np.asarray(t0).ravel(), np.asarray(t1).ravel()
    return float(abs(np.arctan2(t0[0] * t1[1] - t0[1] * t1[0], t0 @ t1)))


# --------------------------------------------------------------------------------------- #
# Construction for a trailing-edge-down deflection
# --------------------------------------------------------------------------------------- #
def _intersect(p: np.ndarray, q: np.ndarray):
    """Intersections of two polylines: (segment index of p, of q, fraction on p, fraction on q)."""
    a0, a1 = p[:-1, None, :], p[1:, None, :]
    b0, b1 = q[None, :-1, :], q[None, 1:, :]
    r, s = a1 - a0, b1 - b0
    den = r[..., 0] * s[..., 1] - r[..., 1] * s[..., 0]
    qp = b0 - a0
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (qp[..., 0] * s[..., 1] - qp[..., 1] * s[..., 0]) / den
        u = (qp[..., 0] * r[..., 1] - qp[..., 1] * r[..., 0]) / den
    hit = (den != 0) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    i, j = np.nonzero(hit)
    return i, j, t[i, j], u[i, j]


def _build_down(L: _Local, hx: float, hy: float, delta: float, prm: GeometryParams):
    """Pieces of the upper and lower surfaces (trailing edge to leading edge, leading edge to
    trailing edge) for a trailing-edge-down deflection ``delta`` (radians, > 0)."""
    phi = -delta
    sA, sL = L.s_at_x(hx, True), L.s_at_x(hx, False)
    A, Lw = L.point(sA)[0], L.point(sL)[0]
    thick = float(A[1] - Lw[1])
    rmin, f, rho = prm.flap_min_arc_radius, prm.flap_fairing_length, prm.flap_fillet_radius
    if thick < 2.0 * rmin:
        raise ValueError(f"the section is {thick:.4g} c thick at hinge_x = {hx:g}, less than twice "
                         f"params.GEOMETRY.flap_min_arc_radius")
    r_open, r_closed = (1.0 - hy) * thick, hy * thick
    if r_open < rmin:
        H, r_open, hy_eff = A.copy(), 0.0, 1.0
    elif r_closed < rmin:
        H, r_closed, hy_eff = Lw.copy(), 0.0, 0.0
    else:
        H, hy_eff = np.array([hx, Lw[1] + hy * thick]), hy
    H[0] = hx
    if sA - f <= 0.0 or L.s_le - sA <= f:
        raise ValueError(f"hinge_x = {hx:g} leaves no room for the fairings "
                         f"(params.GEOMETRY.flap_fairing_length = {f:g} c)")
    rot = (H, phi)
    upper, lower = [], []
    zones_u, zones_l = [], []                      # (first piece index, last piece index) of a zone
    diag = {"hinge_y_effective": hy_eff, "arc_radius": r_open, "fairing_pieces": [],
            "fillet_upper": [], "fillet_lower": [], "sA": sA, "H": H}

    # ---- open side (upper) ------------------------------------------------------------
    if r_open >= rmin:
        sb, sa = sA - f, sA + f
        thA = float(np.arctan2(A[1] - H[1], A[0] - H[0]))
        thB = thA - delta
        B = _rot(A, *rot)[0]
        tA = np.array([-np.sin(thA), np.cos(thA)])
        tB = np.array([-np.sin(thB), np.cos(thB)])
        Pb, Tb = _rot(L.point(sb), *rot)[0], _rotv(L.tangent(sb), phi)[0]
        Pa, Ta = L.point(sa)[0], L.tangent(sa)[0]
        upper.append(_surface_piece(L, 0.0, sb, ZONE_FLAP, rot))
        z0 = len(upper)
        upper.append(_bezier(Pb, Tb, B, tB, ZONE_FLAP, "fairing"))
        upper.append(_arc_piece(H, r_open, thB, thA, ZONE_ARC))
        upper.append(_bezier(A, tA, Pa, Ta, ZONE_MAIN, "fairing"))
        zones_u.append((z0, len(upper) - 1))
        upper.append(_surface_piece(L, sa, L.s_le, ZONE_MAIN))
        diag["arc_length"] = r_open * delta
        diag["fairing_pieces"] = [upper[z0], upper[z0 + 2]]
        diag["thA"], diag["thB"] = thA, thB
    else:
        tin, tout = _rotv(L.tangent(sA), phi)[0], L.tangent(sA)[0]
        alpha = _turn(tin, tout)
        if rho > 0.0 and alpha > 1e-12:
            d = rho * np.tan(0.5 * alpha)
            if d >= min(sA, L.s_le - sA):
                raise ValueError("fillet too large for the flap")
            Pin, Tin = _rot(L.point(sA - d), *rot)[0], _rotv(L.tangent(sA - d), phi)[0]
            Pout, Tout = L.point(sA + d)[0], L.tangent(sA + d)[0]
            upper.append(_surface_piece(L, 0.0, sA - d, ZONE_FLAP, rot))
            z0 = len(upper)
            upper += _split_bezier((Pin, Tin, Pout, Tout), (ZONE_FLAP, ZONE_MAIN))
            zones_u.append((z0, len(upper) - 1))
            upper.append(_surface_piece(L, sA + d, L.s_le, ZONE_MAIN))
            diag["fillet_upper"] += upper[z0:z0 + 2]
        else:
            upper.append(_surface_piece(L, 0.0, sA, ZONE_FLAP, rot))
            upper.append(_surface_piece(L, sA, L.s_le, ZONE_MAIN))
        diag["arc_length"] = 0.0

    # ---- closed side (lower) ----------------------------------------------------------
    if r_closed < rmin:
        sM = sF = sL
    else:
        w = min(0.3, 0.9 * (L.length - sL), 0.9 * (sL - L.s_le))
        # the main surface may extend aft of the cut where the rotated flap meets it (small
        # deflections cross it a little aft of the hinge)
        sm = np.linspace(sL - w, sL + w, 1200)
        sf = np.linspace(sL, sL + w, 600)
        pm, pf = L.point(sm), _rot(L.point(sf), *rot)
        i, j, tt, uu = _intersect(pm, pf)
        if len(i) == 0:
            raise ValueError("the flap does not meet the main surface on the closed side of the "
                             "hinge for this hinge position and deflection")
        k = int(np.argmax(i))
        x0 = [sm[i[k]] + tt[k] * (sm[i[k] + 1] - sm[i[k]]), sf[j[k]] + uu[k] * (sf[j[k] + 1] - sf[j[k]])]
        sol = root(lambda v: (L.point(v[0]) - _rot(L.point(v[1]), *rot))[0], x0, tol=1e-14)
        sM, sF = float(sol.x[0]), float(sol.x[1])
    tin, tout = L.tangent(sM)[0], _rotv(L.tangent(sF), phi)[0]
    alpha = _turn(tin, tout)
    diag["notch_sM_sF"] = (sM, sF)
    if rho > 0.0 and alpha > 1e-12:
        d = rho * np.tan(0.5 * alpha)
        if d >= min(sM - L.s_le, L.length - sF):
            raise ValueError("fillet too large for the flap")
        Pin, Tin = L.point(sM - d)[0], L.tangent(sM - d)[0]
        Pout, Tout = _rot(L.point(sF + d), *rot)[0], _rotv(L.tangent(sF + d), phi)[0]
        lower.append(_surface_piece(L, L.s_le, sM - d, ZONE_MAIN))
        z0 = len(lower)
        lower += _split_bezier((Pin, Tin, Pout, Tout), (ZONE_MAIN, ZONE_FLAP))
        zones_l.append((z0, len(lower) - 1))
        lower.append(_surface_piece(L, sF + d, L.length, ZONE_FLAP, rot))
        diag["fillet_lower"] += lower[z0:z0 + 2]
    else:
        lower.append(_surface_piece(L, L.s_le, sM, ZONE_MAIN))
        lower.append(_surface_piece(L, sF, L.length, ZONE_FLAP, rot))
    return upper, lower, zones_u, zones_l, H, diag


# --------------------------------------------------------------------------------------- #
# Sampling the pieces
# --------------------------------------------------------------------------------------- #
def _sample_surface(pieces: list[_Piece], zones: list[tuple[int, int]], n: int,
                    prm: GeometryParams):
    """``n`` panels over one surface: nodes (n + 1 points) and the zone code of every panel."""
    tabs = [_tables(p) for p in pieces]
    plen = np.array([tb[1][-1] for tb in tabs])
    keep = plen > 1e-10
    pieces = [p for p, kp in zip(pieces, keep) if kp]
    tabs = [tb for tb, kp in zip(tabs, keep) if kp]
    # zones refer to piece indices before dropping empty pieces
    cum_keep = np.cumsum(keep)
    zones = [(int(cum_keep[a] - keep[a]), int(cum_keep[b] - 1)) for a, b in zones]
    plen = np.array([tb[1][-1] for tb in tabs])
    s0 = np.concatenate(([0.0], np.cumsum(plen)))
    total = float(s0[-1])
    # hinge zones: (S start, S end, target spacing)
    zlist = []
    for a, b in zones:
        za, zb = s0[a], s0[b + 1]
        arc = sum(plen[q] for q in range(a, b + 1) if pieces[q].kind == "arc")
        zl = zb - za
        ds_h = zl / prm.flap_zone_panels
        if arc > 0.0:
            ds_h = min(ds_h, arc / prm.flap_arc_min_panels)
        ds_h = max(ds_h, zl / (4.0 * prm.flap_zone_panels))
        zlist.append((za, zb, ds_h))
    g = prm.flap_grading - 1.0
    smid = [s0[i] + 0.5 * (tb[1][:-1] + tb[1][1:]) for i, tb in enumerate(tabs)]
    dl = [np.diff(tb[1]) for tb in tabs]

    def ds_zone(S):
        out = np.full_like(S, np.inf)
        for za, zb, dh in zlist:
            out = np.minimum(out, dh + g * np.maximum(np.maximum(za - S, S - zb), 0.0))
        return out

    dz = [ds_zone(s) for s in smid]

    def counts(k):
        out = []
        for s, d, z in zip(smid, dl, dz):
            u = s / total
            ds = np.minimum(k * np.sqrt(u * (1.0 - u)), z)
            out.append(np.concatenate(([0.0], np.cumsum(d / ds))))
        return out

    lo, hi = 1e-9, 1e3
    if sum(c[-1] for c in counts(hi)) > n:
        raise ValueError(f"{n} panels per surface are too few for the hinge zones")
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if sum(c[-1] for c in counts(mid)) > n:
            lo = mid
        else:
            hi = mid
    cum = counts(np.sqrt(lo * hi))
    real = np.array([c[-1] for c in cum])
    cnt = np.maximum(1, np.rint(real).astype(int))
    while cnt.sum() != n:
        if cnt.sum() < n:
            cnt[int(np.argmax(real - cnt))] += 1
        else:
            cand = np.where(cnt > 1, cnt - real, -np.inf)
            cnt[int(np.argmax(cand))] -= 1
    nodes, zone_codes = [], []
    for p, (t, l), c, m in zip(pieces, tabs, cum, cnt):
        want = c[-1] * np.arange(m) / m
        lpos = np.interp(want, c, l)
        tpos = np.interp(lpos, l, t)
        nodes.append(p.f(tpos))
        zone_codes += [p.zone] * m
    nodes.append(pieces[-1].f(np.array([1.0])))
    return np.vstack(nodes), np.array(zone_codes, dtype=np.int8), (pieces, plen, cnt)


def _check_simple(c: np.ndarray) -> None:
    """Raise if the polyline crosses itself (non-adjacent segments only)."""
    i, j, t, u = _intersect(c, c)
    bad = (j > i + 1) & ~((i == 0) & (j == len(c) - 2))
    eps = 1e-12
    bad &= (t > eps) & (t < 1 - eps) & (u > eps) & (u < 1 - eps)
    if np.any(bad):
        raise ValueError("the flapped contour crosses itself; reduce the deflection, move the "
                         "hinge, or reduce the fairing or fillet size")


def _dist_to_polyline(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    a, b = poly[:-1][None], poly[1:][None]
    p = pts[:, None, :]
    ab = b - a
    tt = np.clip(np.sum((p - a) * ab, axis=2) / np.maximum(np.sum(ab * ab, axis=2), 1e-300), 0, 1)
    d = p - (a + tt[..., None] * ab)
    return np.sqrt(np.min(np.sum(d * d, axis=2), axis=1))


def _deviation(L, rot, diag, delta) -> dict:
    """Largest distance (chords) of the fairings and fillets from the unfaired contour: surface,
    arc and rotated surface on the open side; the sharp notch on the closed side."""
    H = diag["H"]
    ts = np.linspace(0, 1, 4000)
    sA = diag["sA"]
    parts = [_rot(L.point(sA * ts), *rot)]
    if diag["arc_radius"] > 0.0:
        th = diag["thB"] + (diag["thA"] - diag["thB"]) * ts
        parts.append(H + diag["arc_radius"] * np.column_stack([np.cos(th), np.sin(th)]))
    parts.append(L.point(sA + (L.s_le - sA) * ts))
    ref_open = np.vstack(parts)
    sM, sF = diag["notch_sM_sF"]
    ref_closed = np.vstack([L.point(L.s_le + (sM - L.s_le) * ts),
                            _rot(L.point(sF + (L.length - sF) * ts), *rot)])
    tt = np.linspace(0, 1, 60)

    def worst(pieces, ref):
        return max([float(np.max(_dist_to_polyline(pc.f(tt), ref))) for pc in pieces] + [0.0])
    return {"fairing_deviation": worst(diag["fairing_pieces"], ref_open),
            "fillet_deviation": max(worst(diag["fillet_upper"], ref_open),
                                    worst(diag["fillet_lower"], ref_closed))}


# --------------------------------------------------------------------------------------- #
# Public construction
# --------------------------------------------------------------------------------------- #
def undeflected_zones(coords: np.ndarray, hinge_x: float) -> np.ndarray:
    """Zones of an undeflected section: panels whose midpoint lies aft of ``hinge_x`` are the flap."""
    L = _Local(coords)
    mid = 0.5 * (np.asarray(coords)[:-1] + np.asarray(coords)[1:])
    d = mid - L.le
    xl = (d @ L.e) / L.chord
    return np.where(xl > hinge_x, ZONE_FLAP, ZONE_MAIN).astype(np.int8)


def hinge_point(coords: np.ndarray, hinge_x: float, hinge_y: float = 0.5) -> np.ndarray:
    """Hinge point of a section (its own frame): fraction ``hinge_y`` of the thickness at ``hinge_x``."""
    L = _Local(coords)
    A, Lw = L.point(L.s_at_x(hinge_x, True))[0], L.point(L.s_at_x(hinge_x, False))[0]
    h = np.array([[hinge_x, Lw[1] + hinge_y * (A[1] - Lw[1])]])
    return geo._from_chord_frame(h, L.frame)[0]


def make_flap(coords: np.ndarray, hinge_x: float, deflection_deg: float, hinge_y: float = 0.5,
              num_panels: int | None = None, params: GeometryParams = GEOMETRY):
    """Flapped section: ``(coordinates, FlapInfo)``.

    ``coords`` is the undeflected section (Selig order, any position, rotation and scale);
    ``hinge_x`` the chordwise hinge position (fraction of its chord), ``hinge_y`` the fraction of
    the local thickness (0 lower, 1 upper surface), ``deflection_deg`` positive trailing edge
    down, from -30 to +30 degrees. ``num_panels`` defaults to the number of panels of ``coords``.
    A deflection of 0 returns ``coords`` unchanged.
    """
    c = np.array(coords, dtype=float)
    if not 0.0 < hinge_x < 1.0:
        raise ValueError(f"hinge_x must be inside the chord, got {hinge_x!r}")
    if not 0.0 <= hinge_y <= 1.0:
        raise ValueError(f"hinge_y must be between 0 and 1, got {hinge_y!r}")
    if abs(deflection_deg) > 30.0 + 1e-9:
        raise ValueError(f"the deflection must be within +-30 degrees, got {deflection_deg!r}")
    n = len(c) - 1 if num_panels is None else int(num_panels)
    hinge = hinge_point(c, hinge_x, hinge_y)
    if deflection_deg == 0.0:
        out = c if num_panels is None or n == len(c) - 1 else geo.repanel_cosine(c, n)
        info = FlapInfo(_ro(c), float(hinge_x), float(hinge_y), 0.0, _ro(hinge),
                        undeflected_zones(out, hinge_x), params, {"deflection": 0.0})
        return out, info
    mirror = deflection_deg < 0.0
    L = _Local(c, mirror=mirror)
    hy = 1.0 - hinge_y if mirror else hinge_y
    delta = np.radians(abs(deflection_deg))
    upper, lower, zu, zl, H, diag = _build_down(L, hinge_x, hy, delta, params)
    h = n // 2
    nu, zu_codes, (pu, plu, cu) = _sample_surface(upper, zu, h, params)
    nl, zl_codes, (pl, pll, cl) = _sample_surface(lower, zl, n - h, params)
    local = np.vstack([nu, nl[1:]])
    zones = np.concatenate([zu_codes, zl_codes])
    if mirror:
        local = local[::-1].copy()
        local[:, 1] = -local[:, 1]
        zones = zones[::-1].copy()
        H = np.array([H[0], -H[1]])
    out = geo._from_chord_frame(local, L.frame)
    _check_simple(out)
    hinge_abs = geo._from_chord_frame(H[None, :], L.frame)[0]
    rot = (diag["H"], -delta)
    dev = _deviation(L, rot, diag, delta)
    narc = int(np.sum(zones == ZONE_ARC))
    diagn = {"deflection": float(deflection_deg), "hinge_y_effective": diag["hinge_y_effective"]
             if not mirror else 1.0 - diag["hinge_y_effective"], "arc_radius": diag["arc_radius"],
             "arc_length": diag["arc_length"], "arc_panels": narc, **dev}
    info = FlapInfo(_ro(c), float(hinge_x), float(hinge_y), float(deflection_deg), _ro(hinge_abs),
                    zones, params, diagn)
    return out, info


def _ro(a: np.ndarray) -> np.ndarray:
    a = np.array(a, dtype=float)
    a.setflags(write=False)
    return a
