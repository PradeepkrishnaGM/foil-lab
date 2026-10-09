# src/foil_lab/generators.py

import numpy as np

# Last coefficient of the NACA 4-digit thickness polynomial. "open" is the original
# definition (Jacobs, Ward and Pinkerton, NACA Report 460, 1933; Abbott and von Doenhoff,
# "Theory of Wing Sections", 1959), which leaves a trailing-edge gap of about 1 % of the
# thickness. "closed" is the common modification that closes the trailing edge to a point.
_NACA_TE_COEFF = {"open": -0.1015, "closed": -0.1036}


def _naca_thickness(x: np.ndarray, t: float, trailing_edge: str) -> np.ndarray:
    """NACA 4/5-digit half-thickness distribution for thickness ratio ``t``.

    ``trailing_edge`` is ``"closed"`` (sharp, coefficient -0.1036) or ``"open"``
    (NACA Report 460, coefficient -0.1015).
    """
    if trailing_edge not in _NACA_TE_COEFF:
        raise ValueError(f"trailing_edge must be 'open' or 'closed', got {trailing_edge!r}")
    a4 = _NACA_TE_COEFF[trailing_edge]
    return 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 + a4 * x**4)


def generate_naca4(naca_string: str, num_panels: int = 120,
                   trailing_edge: str = "open") -> np.ndarray:
    """
    Generates a NACA 4-digit airfoil using cosine spacing.
    Expects a 4-character string (e.g., '2412', '0012').

    ``trailing_edge`` selects the thickness definition: ``"open"`` (default since v2.0; NACA
    Report 460, as used by XFOIL's ``NACA`` command) or ``"closed"`` (sharp trailing edge;
    output unchanged from v1.3.1).
    """
    if len(naca_string) != 4 or not naca_string.isdigit():
        raise ValueError("NACA 4-digit string must be exactly 4 numbers (e.g., '2412').")

    # Parse the parameters from the string
    m = int(naca_string[0]) / 100.0       # Max camber
    p = int(naca_string[1]) / 10.0        # Position of max camber
    t = int(naca_string[2:4]) / 100.0     # Max thickness

    # Cosine spacing for the x-coordinates (0 to 1)
    num_points = (num_panels // 2) + 1
    beta = np.linspace(0, np.pi, num_points)
    x = 0.5 * (1 - np.cos(beta))

    # Thickness distribution (closed or open trailing edge)
    yt = _naca_thickness(x, t, trailing_edge)

    # Calculate Camber Line (yc) and its gradient (dyc/dx)
    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)

    if m > 0:
        # Forward of max camber
        front_idx = x <= p
        yc[front_idx] = (m / p**2) * (2 * p * x[front_idx] - x[front_idx]**2)
        dyc_dx[front_idx] = (2 * m / p**2) * (p - x[front_idx])

        # Aft of max camber
        back_idx = x > p
        yc[back_idx] = (m / (1 - p)**2) * ((1 - 2 * p) + 2 * p * x[back_idx] - x[back_idx]**2)
        dyc_dx[back_idx] = (2 * m / (1 - p)**2) * (p - x[back_idx])

    # Calculate the angle of the camber line
    theta = np.arctan(dyc_dx)

    # Calculate Upper and Lower Surface Coordinates
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)

    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    # Reassemble in Selig format (TE Upper -> LE -> TE Lower)
    # Note: We reverse the upper surface arrays to go from TE to LE
    upper_surface = np.column_stack((xu[::-1], yu[::-1]))
    lower_surface = np.column_stack((xl[1:], yl[1:])) # Skip LE duplicate

    return np.vstack((upper_surface, lower_surface))
    
def generate_naca5(naca_string: str, num_panels: int = 120,
                   trailing_edge: str = "open") -> np.ndarray:
    """
    Generates a NACA 5-digit airfoil (e.g., '23012').

    ``trailing_edge`` selects the thickness definition: ``"open"`` (default since v2.0; NACA
    Report 460) or ``"closed"`` (sharp trailing edge; output unchanged from v1.3.1).
    """
    if len(naca_string) != 5 or not naca_string.isdigit():
        raise ValueError("NACA 5-digit string must be exactly 5 numbers (e.g., '23012').")

    # Parse standard parameters
    cl_design = int(naca_string[0]) * 0.15
    t = int(naca_string[3:5]) / 100.0

    # Lookup table for standard 5-digit camber profiles (First 3 digits)
    # Maps 'identifier' -> (m, k1)
    camber_table = {
        '210': (0.0580, 361.400),
        '220': (0.1260, 51.640),
        '230': (0.2025, 15.957),
        '240': (0.2900, 6.643),
        '250': (0.3910, 3.230)
    }
    
    identifier = naca_string[0:3]
    if identifier not in camber_table:
        raise ValueError(f"Camber profile '{identifier}' is not a standard, supported 5-digit prefix.")
        
    m, k1 = camber_table[identifier]

    # Cosine spacing
    num_points = (num_panels // 2) + 1
    beta = np.linspace(0, np.pi, num_points)
    x = 0.5 * (1 - np.cos(beta))

    # Standard 4-digit/5-digit thickness distribution (closed or open trailing edge)
    yt = _naca_thickness(x, t, trailing_edge)

    # 5-Digit Camber Line
    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)

    front_idx = x <= m
    back_idx = x > m

    yc[front_idx] = (k1 / 6.0) * (x[front_idx]**3 - 3 * m * x[front_idx]**2 + m**2 * (3 - m) * x[front_idx])
    dyc_dx[front_idx] = (k1 / 6.0) * (3 * x[front_idx]**2 - 6 * m * x[front_idx] + m**2 * (3 - m))

    yc[back_idx] = (k1 * m**3 / 6.0) * (1 - x[back_idx])
    dyc_dx[back_idx] = -(k1 * m**3 / 6.0)

    # Apply scaling for design lift coefficient
    # The standard equations are for Cl = 0.3 (the '2' in 23012). We scale it linearly.
    scale_factor = cl_design / 0.3
    yc *= scale_factor
    dyc_dx *= scale_factor

    theta = np.arctan(dyc_dx)

    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    upper_surface = np.column_stack((xu[::-1], yu[::-1]))
    lower_surface = np.column_stack((xl[1:], yl[1:]))

    return np.vstack((upper_surface, lower_surface))


# =========================================================================== #
# Advanced generators: PARSEC, supersonic (biconvex / wedge)
# =========================================================================== #

def generate_parsec(params: dict, num_panels: int = 160) -> np.ndarray:
    """Generate an airfoil from the 11-parameter PARSEC parameterization.

    PARSEC describes each surface by a 6-term power series in sqrt(x); the 12
    coefficients (6 per surface) are fixed by 11 geometric parameters plus the
    closure of the system. This representation is popular for gradient-based
    aerodynamic optimization because the parameters are physically meaningful.

    Expected keys in ``params`` (all floats):
        rle        leading-edge radius
        x_up, y_up, yxx_up   crest location, height, curvature (upper)
        x_lo, y_lo, yxx_lo   crest location, height, curvature (lower)
        y_te       trailing-edge y-offset (camber at TE)
        dy_te      trailing-edge thickness (half-gap)
        ang_te     trailing-edge direction angle (deg)
        ang_wedge  trailing-edge wedge half-angle (deg)
    """
    req = ["rle", "x_up", "y_up", "yxx_up", "x_lo", "y_lo", "yxx_lo",
           "y_te", "dy_te", "ang_te", "ang_wedge"]
    missing = [k for k in req if k not in params]
    if missing:
        raise ValueError(f"PARSEC missing parameters: {missing}")

    a_te = np.deg2rad(params["ang_te"])
    a_w = np.deg2rad(params["ang_wedge"])

    def surface_coeffs(x_c, y_c, yxx_c, y_te, dy_te, ang, rle, upper):
        # Solve the 6x6 linear system for the PARSEC power-series coefficients.
        # Series: y(x) = sum_{n=1..6} a_n * x^((2n-1)/2)
        sgn = 1.0 if upper else -1.0
        # TE point
        y_te_s = y_te + sgn * dy_te / 2.0
        # surface TE direction angle
        ang_s = ang - sgn * a_w
        # Build constraints (Sobieczky's standard PARSEC system).
        exps = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
        A = np.zeros((6, 6))
        b = np.zeros(6)
        # 1) y(1) = y_te
        A[0] = 1.0 ** exps
        b[0] = y_te_s
        # 2) y(x_c) = y_c
        A[1] = x_c ** exps
        b[1] = y_c
        # 3) y'(1) = tan(ang_s)
        A[2] = exps * 1.0 ** (exps - 1)
        b[2] = np.tan(ang_s)
        # 4) y'(x_c) = 0  (crest)
        A[3] = exps * x_c ** (exps - 1)
        b[3] = 0.0
        # 5) y''(x_c) = yxx_c
        A[4] = exps * (exps - 1) * x_c ** (exps - 2)
        b[4] = yxx_c
        # 6) LE radius: a_1 = sgn*sqrt(2*rle)
        A[5] = np.array([1, 0, 0, 0, 0, 0], dtype=float)
        b[5] = sgn * np.sqrt(2.0 * rle)
        return np.linalg.solve(A, b), exps

    c_up, exps = surface_coeffs(params["x_up"], params["y_up"], params["yxx_up"],
                                params["y_te"], params["dy_te"], a_te,
                                params["rle"], upper=True)
    c_lo, _ = surface_coeffs(params["x_lo"], params["y_lo"], params["yxx_lo"],
                             params["y_te"], params["dy_te"], a_te,
                             params["rle"], upper=False)

    num_points = (num_panels // 2) + 1
    beta = np.linspace(0, np.pi, num_points)
    x = 0.5 * (1 - np.cos(beta))

    def evals(coeffs):
        return np.sum([coeffs[i] * x ** exps[i] for i in range(6)], axis=0)

    yu = evals(c_up)
    yl = evals(c_lo)

    upper = np.column_stack((x[::-1], yu[::-1]))
    lower = np.column_stack((x[1:], yl[1:]))
    return np.vstack((upper, lower))


def generate_biconvex(thickness: float = 0.06, num_panels: int = 120) -> np.ndarray:
    """Symmetric biconvex (double-circular-arc) supersonic section.

    Each surface is a parabolic arc y = +/- 2 t x (1 - x), giving a sharp LE and
    TE -- a canonical supersonic profile for linearized/shock-expansion studies.
    """
    num_points = (num_panels // 2) + 1
    beta = np.linspace(0, np.pi, num_points)
    x = 0.5 * (1 - np.cos(beta))
    y = 2.0 * thickness * x * (1.0 - x)
    upper = np.column_stack((x[::-1], y[::-1]))
    lower = np.column_stack((x[1:], -y[1:]))
    return np.vstack((upper, lower))


def generate_wedge(thickness: float = 0.06, num_panels: int = 120) -> np.ndarray:
    """Symmetric double-wedge (diamond) supersonic section.

    Straight ramps rise from a sharp LE to max thickness at mid-chord, then fall
    to a sharp TE. ``thickness`` is t/c (max half-thickness = thickness/2).
    """
    num_points = (num_panels // 2) + 1
    beta = np.linspace(0, np.pi, num_points)
    x = 0.5 * (1 - np.cos(beta))
    half = thickness / 2.0
    y = np.where(x <= 0.5, half * (x / 0.5), half * ((1.0 - x) / 0.5))
    upper = np.column_stack((x[::-1], y[::-1]))
    lower = np.column_stack((x[1:], -y[1:]))
    return np.vstack((upper, lower))
