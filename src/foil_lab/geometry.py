import numpy as np
from scipy.interpolate import PchipInterpolator

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

def repanel_cosine(coords: np.ndarray, num_panels: int = 100) -> np.ndarray:
    """
    Repanels the airfoil using cosine spacing to cluster points at the LE and TE.
    Uses a monotonic cubic interpolator to prevent spline wiggles.
    """
    # Number of points is num_panels + 1. Split roughly in half for upper/lower
    num_points_per_surface = (num_panels // 2) + 1
    
    upper, lower = split_surfaces(coords)
    
    # Generate cosine-spaced X coordinates: 0.0 (LE) to 1.0 (TE)
    beta = np.linspace(0, np.pi, num_points_per_surface)
    x_new = 0.5 * (1 - np.cos(beta))
    
    # Helper function to safely interpolate surfaces
    def interpolate_surface(surface_coords):
        # Sort by X to ensure strictly increasing arrays for interpolation
        sort_idx = np.argsort(surface_coords[:, 0])
        x_sorted = surface_coords[sort_idx, 0]
        y_sorted = surface_coords[sort_idx, 1]
        
        # Remove duplicate X values (sometimes occurs exactly at the LE)
        unique_idx = np.unique(x_sorted, return_index=True)[1]
        x_unique = x_sorted[unique_idx]
        y_unique = y_sorted[unique_idx]
        
        # PchipInterpolator is used instead of CubicSpline to prevent overshoot/wiggles
        interpolator = PchipInterpolator(x_unique, y_unique)
        return interpolator(x_new)
        
    y_upper_new = interpolate_surface(upper)
    y_lower_new = interpolate_surface(lower)
    
    # Reassemble in Selig format: TE Upper (x=1) -> LE (x=0) -> TE Lower (x=1)
    upper_repanelled = np.column_stack((x_new[::-1], y_upper_new[::-1]))
    
    # Skip the first point of the lower surface (x=0) to avoid duplicating the LE point
    lower_repanelled = np.column_stack((x_new[1:], y_lower_new[1:]))
    
    return np.vstack((upper_repanelled, lower_repanelled))

# Add to the bottom of src/foil_lab/geometry.py
def deflect_flap(coords: np.ndarray, hinge_x: float, deflection_deg: float) -> np.ndarray:
    """
    Simulates a simple plain flap by rotating all coordinates aft of the 
    hinge_x around the camber line at the hinge location.
    Downwards deflection is positive.
    """
    if hinge_x <= 0.0 or hinge_x >= 1.0:
        return coords # Hinge must be inside the airfoil
        
    upper, lower = split_surfaces(coords)
    
    # 1. Find the Y coordinate of the hinge on the camber line
    # We interpolate the upper and lower surfaces at hinge_x
    y_up_hinge = np.interp(hinge_x, upper[::-1, 0], upper[::-1, 1])
    y_low_hinge = np.interp(hinge_x, lower[:, 0], lower[:, 1])
    y_hinge = (y_up_hinge + y_low_hinge) / 2.0
    
    # 2. Define the rotation matrix
    angle_rad = np.deg2rad(-deflection_deg) # Negative because positive flap is trailing edge DOWN
    cos_val, sin_val = np.cos(angle_rad), np.sin(angle_rad)
    rot_matrix = np.array([
        [cos_val, -sin_val],
        [sin_val,  cos_val]
    ])
    
    new_coords = np.copy(coords)
    
    # 3. Rotate all points aft of the hinge
    for i in range(len(new_coords)):
        x, y = new_coords[i, 0], new_coords[i, 1]
        if x >= hinge_x:
            # Translate hinge to origin, rotate, translate back
            shifted = np.array([x - hinge_x, y - y_hinge])
            rotated = np.dot(shifted, rot_matrix.T)
            new_coords[i, 0] = rotated[0] + hinge_x
            new_coords[i, 1] = rotated[1] + y_hinge
            
    return new_coords

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
