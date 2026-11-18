import numpy as np
from scipy.interpolate import CubicSpline

def normalize_airfoil(coords: np.ndarray) -> np.ndarray:
    """
    Normalizes airfoil coordinates:
    - Leading edge at (0,0)
    - Trailing edge on the x-axis
    - Unit chord length (1.0)
    """
    # Identify Leading Edge (LE) as the point with the minimum X coordinate
    le_idx = np.argmin(coords[:, 0])
    le_point = coords[le_idx]
    
    # Shift geometry so LE is exactly at the origin
    shifted = coords - le_point
    
    # Trailing Edge (TE) is approximated as the average of the first and last points
    te_point = (shifted[0] + shifted[-1]) / 2.0
    chord = np.linalg.norm(te_point)
    
    # Calculate angle to rotate the TE onto the x-axis
    angle = -np.arctan2(te_point[1], te_point[0])
    cos_val, sin_val = np.cos(angle), np.sin(angle)
    
    rotation_matrix = np.array([
        [cos_val, -sin_val],
        [sin_val,  cos_val]
    ])
    
    rotated = np.dot(shifted, rotation_matrix.T)
    
    # Scale coordinates by the chord length
    normalized = rotated / chord
    
    return normalized

def split_surfaces(coords: np.ndarray):
    """
    Splits coordinates into upper and lower surfaces based on the LE index.
    Assumes Selig format (TE Upper -> LE -> TE Lower).
    """
    le_idx = np.argmin(coords[:, 0])
    upper = coords[:le_idx+1]
    lower = coords[le_idx:]
    return upper, lower

class _ArcSpline:
    """Cubic spline of a Selig-ordered section in arc length, with its leading edge.

    The leading edge is the point of the spline farthest from the trailing-edge midpoint (the
    mean of the first and last points), found by root finding on the derivative of the squared
    distance. Arc length and that distance do not depend on position, rotation or scale, so
    everything built on this is independent of chord length and position.
    """

    def __init__(self, coords: np.ndarray):
        c = np.asarray(coords, dtype=float)
        seg = np.hypot(*np.diff(c, axis=0).T)
        keep = np.concatenate(([True], seg > 0.0))       # drop repeated points
        c = c[keep]
        s = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(c, axis=0).T))))
        self.nodes, self.s = c, s
        self.x, self.y = CubicSpline(s, c[:, 0]), CubicSpline(s, c[:, 1])
        self.length = float(s[-1])
        self.te_mid = 0.5 * (c[0] + c[-1])
        self.s_le = self._leading_edge()

    def point(self, s):
        return np.column_stack([self.x(s), self.y(s)])

    def _leading_edge(self) -> float:
        from scipy.optimize import brentq
        from .params import GEOMETRY
        m = GEOMETRY.le_search_samples * len(self.nodes)
        ss = np.linspace(0.0, self.length, m)
        d2 = (self.x(ss) - self.te_mid[0]) ** 2 + (self.y(ss) - self.te_mid[1]) ** 2
        k = int(np.argmax(d2))
        lo, hi = ss[max(k - 1, 0)], ss[min(k + 1, m - 1)]

        def f(t):
            return ((self.x(t) - self.te_mid[0]) * self.x(t, 1)
                    + (self.y(t) - self.te_mid[1]) * self.y(t, 1))
        if f(lo) * f(hi) < 0.0:
            return float(brentq(f, lo, hi, xtol=1e-15 * self.length, rtol=8.9e-16))
        return float(ss[k])

    def chord_frame(self):
        """(leading edge, unit chord direction, chord length)."""
        le = self.point(self.s_le)[0]
        d = self.te_mid - le
        chord = float(np.hypot(*d))
        return le, d / chord, chord


def repanel_cosine(coords: np.ndarray, num_panels: int | None = None) -> np.ndarray:
    """Repanel a section by arc length with cosine clustering at both edges.

    A cubic spline in arc length is fitted through the Selig-ordered points; the leading edge
    is the spline point farthest from the trailing-edge midpoint. Each surface (trailing edge
    to leading edge, leading edge to trailing edge) gets ``num_panels // 2`` panels (the lower
    surface the rest) with cosine spacing in arc length, which clusters nodes at the leading and
    trailing edges. The end points are kept exactly, so the trailing-edge gap is unchanged. The
    result does not depend on chord length, position or rotation of the input.
    ``num_panels`` defaults to ``params.GEOMETRY.num_panels``.
    """
    from .params import GEOMETRY
    n = GEOMETRY.num_panels if num_panels is None else int(num_panels)
    sp = _ArcSpline(coords)
    h = n // 2
    t1 = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, h + 1)))
    t2 = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n - h + 1)))
    s = np.concatenate([sp.s_le * t1, sp.s_le + (sp.length - sp.s_le) * t2[1:]])
    out = sp.point(s)
    out[0], out[-1] = sp.nodes[0], sp.nodes[-1]
    return out


def _to_chord_frame(coords: np.ndarray, sp: "_ArcSpline"):
    le, e, chord = sp.chord_frame()
    nrm = np.array([-e[1], e[0]])
    d = np.asarray(coords, dtype=float) - le
    return np.column_stack([d @ e, d @ nrm]) / chord, (le, e, nrm, chord)


def _from_chord_frame(local: np.ndarray, frame) -> np.ndarray:
    le, e, nrm, chord = frame
    return le + chord * (local[:, :1] * e + local[:, 1:] * nrm)


def _other_surface_y(sp: "_ArcSpline", frame, x_local: float, upper: bool) -> float:
    """y (chord frame) of the opposite surface at chordwise position ``x_local``."""
    from scipy.optimize import brentq
    le, e, nrm, chord = frame
    a, b = (sp.s_le, sp.length) if upper else (0.0, sp.s_le)

    def xl(t):
        return float(((sp.point(t)[0] - le) @ e) / chord)
    xa, xb = xl(a), xl(b)
    lo_x, hi_x = min(xa, xb), max(xa, xb)
    if x_local <= lo_x:
        t = a if xa <= xb else b
    elif x_local >= hi_x:
        t = b if xb >= xa else a
    else:
        t = brentq(lambda q: xl(q) - x_local, a, b, xtol=1e-15 * sp.length, rtol=8.9e-16)
    return float(((sp.point(t)[0] - le) @ nrm) / chord)


def scale_thickness_camber(coords: np.ndarray, thickness: float = 1.0,
                           camber: float = 1.0) -> np.ndarray:
    """Scale the thickness and the camber of a section in its own chord frame.

    The chord runs from the leading edge (see :func:`repanel_cosine`) to the trailing-edge
    midpoint. At every node the opposite surface is evaluated at the same chordwise position on
    the arc-length spline; camber c = (y_u + y_l)/2 and thickness t = y_u - y_l, and the node
    moves to c' + t'/2 (upper) or c' - t'/2 (lower) with c' = camber * c and t' = thickness * t.
    Chordwise positions are kept, so factors of 1 return the input (to round-off).
    """
    c = np.asarray(coords, dtype=float)
    sp = _ArcSpline(c)
    local, frame = _to_chord_frame(c, sp)
    s_nodes = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(c, axis=0).T))))
    out = local.copy()
    for i, (xl, yl) in enumerate(local):
        upper = s_nodes[i] <= sp.s_le
        yo = _other_surface_y(sp, frame, xl, upper)
        yu, ylo = (yl, yo) if upper else (yo, yl)
        cam, thk = 0.5 * (yu + ylo), yu - ylo
        out[i, 1] = camber * cam + (0.5 if upper else -0.5) * thickness * thk
    # The two trailing-edge nodes are one station (the trailing-edge midpoint, x' = 1): camber
    # and thickness there come from the pair itself, so a zero-camber result keeps its
    # trailing-edge midpoint on the chord line (DECISIONS.md, M2 step 4).
    if len(local) > 2 and not np.array_equal(local[0], local[-1]):
        cam_te, thk_te = 0.5 * (local[0, 1] + local[-1, 1]), local[0, 1] - local[-1, 1]
        out[0, 1] = camber * cam_te + 0.5 * thickness * thk_te
        out[-1, 1] = camber * cam_te - 0.5 * thickness * thk_te
    return _from_chord_frame(out, frame)


def trailing_edge_gap(coords: np.ndarray) -> float:
    """Distance between the first and last points (zero for a closed trailing edge)."""
    c = np.asarray(coords, dtype=float)
    return float(np.hypot(*(c[0] - c[-1])))


def set_trailing_edge_gap(coords: np.ndarray, gap: float,
                          blend_start: float | None = None) -> np.ndarray:
    """Return the section with trailing-edge gap ``gap`` (same length units as the input).

    The thickness is changed only aft of ``blend_start`` (fraction of chord, default
    ``params.GEOMETRY.te_gap_blend_start``) by d(x) = D ((x - x_b) / (1 - x_b))^2, split equally
    between the surfaces normal to the chord, so the shape ahead of x_b is unchanged and the
    change starts with zero slope. D is chosen so that the new gap equals ``gap``. A gap above
    ``params.GEOMETRY.te_gap_warn`` of the chord gives a warning.
    """
    import warnings
    from scipy.optimize import brentq
    from .params import GEOMETRY
    xb = GEOMETRY.te_gap_blend_start if blend_start is None else float(blend_start)
    c = np.asarray(coords, dtype=float)
    sp = _ArcSpline(c)
    local, frame = _to_chord_frame(c, sp)
    chord = frame[3]
    if gap < 0.0:
        raise ValueError("gap must be non-negative")
    if gap > GEOMETRY.te_gap_warn * chord:
        warnings.warn(f"trailing-edge gap {gap / chord:.4f} c is above "
                      f"{GEOMETRY.te_gap_warn:.3g} c (params.GEOMETRY.te_gap_warn)", stacklevel=2)
    s_nodes = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(c, axis=0).T))))
    side = np.where(s_nodes <= sp.s_le, 0.5, -0.5)
    w = np.where(local[:, 0] > xb, ((local[:, 0] - xb) / (1.0 - xb)) ** 2, 0.0)
    # the two trailing-edge nodes are one station (x' = 1): equal and opposite changes, so the
    # trailing-edge midpoint does not move when only the gap changes (DECISIONS.md, M2 step 4)
    w[0] = w[-1] = 1.0

    def shaped(delta):
        out = local.copy()
        out[:, 1] = local[:, 1] + side * w * delta
        return out

    def gap_of(delta):
        o = shaped(delta)
        return float(np.hypot(*(o[0] - o[-1]))) - gap / chord
    g0 = gap_of(0.0)
    if abs(g0) <= 1e-15:
        return c.copy()
    # the gap is |opening| and so not monotonic in delta: search each side of zero separately
    big = 4.0 * (abs(g0) + gap / chord) + 1.0
    delta = None
    for lo, hi in ((0.0, big), (-big, 0.0)):
        if gap_of(lo) * gap_of(hi) <= 0.0:
            delta = brentq(gap_of, lo, hi, xtol=1e-16, rtol=8.9e-16)
            break
    if delta is None:
        raise ValueError("could not reach the requested trailing-edge gap")
    return _from_chord_frame(shaped(delta), frame)


def translate(coords: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Shifts the geometry by dx and dy."""
    return coords + np.array([dx, dy])

def scale(coords: np.ndarray, factor: float) -> np.ndarray:
    """Scales the geometry uniformly."""
    if factor <= 0:
        raise ValueError("Scale factor must be strictly positive.")
    return coords * factor

def rotate(coords: np.ndarray, angle_deg: float, pivot: tuple = (0.0, 0.0)) -> np.ndarray:
    """
    Rotates the geometry around a specified pivot point.
    Positive angle pitches the nose UP (aerodynamic convention).
    """
    # Negative mathematically, so positive angle pitches nose UP (trailing edge down)
    angle_rad = np.deg2rad(-angle_deg) 
    
    cos_val, sin_val = np.cos(angle_rad), np.sin(angle_rad)
    rot_matrix = np.array([
        [cos_val, -sin_val],
        [sin_val,  cos_val]
    ])
    
    pivot_arr = np.array(pivot)
    shifted = coords - pivot_arr
    rotated = np.dot(shifted, rot_matrix.T)
    
    return rotated + pivot_arr


def smooth_coordinates(coords: np.ndarray, strength: float = 0.5,
                       iterations: int = 1) -> np.ndarray:
    """Remove numerical bumps from imported coordinates with a weighted average.

    Each interior point is blended toward the midpoint of its neighbours:

        y_i <- (1 - strength) * y_i + strength * 0.5 * (y_{i-1} + y_{i+1})

    and likewise for x. This is a fast, NumPy-vectorized Laplacian smoothing
    pass that damps high-frequency wiggles in raw `.dat` data which can
    destabilize the panel solver. Endpoints (the trailing edge) are preserved.

    Parameters
    ----------
    strength : 0..1 blend factor (0 = no change, 1 = full neighbour average).
    iterations : number of smoothing passes.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    out = coords.astype(float).copy()
    for _ in range(max(1, iterations)):
        prev = out.copy()
        # Vectorized neighbour average for interior points.
        neighbour_avg = 0.5 * (prev[:-2] + prev[2:])
        out[1:-1] = (1.0 - strength) * prev[1:-1] + strength * neighbour_avg
    return out


def bodies_overlap(bodies, margin: float = 1e-3) -> bool:
    """Cheap bounding-box overlap test for a list of bodies.

    Panel methods require non-intersecting bodies; overlapping geometry produces
    nonsensical forces. This conservative AABB check flags obvious overlaps so
    callers (e.g. the GUI) can warn the user. It can miss non-overlapping AABBs
    that still intersect, and flag touching-but-valid configs, so it is advisory.
    """
    import numpy as np
    boxes = []
    for b in bodies:
        b = np.asarray(b)
        boxes.append((b[:, 0].min(), b[:, 0].max(), b[:, 1].min(), b[:, 1].max()))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            ax0, ax1, ay0, ay1 = boxes[i]
            bx0, bx1, by0, by1 = boxes[j]
            if (ax0 < bx1 - margin and bx0 < ax1 - margin and
                    ay0 < by1 - margin and by0 < ay1 - margin):
                return True
    return False
