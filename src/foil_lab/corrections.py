# src/foil_lab/corrections.py

import numpy as np

def karman_tsien_correction(cp_incompressible: np.ndarray, mach: float) -> np.ndarray:
    """
    Applies the Kármán-Tsien compressibility correction to an array of 
    incompressible pressure coefficients.
    Valid for subsonic flows (Mach < 0.7).
    """
    if mach <= 0.0 or mach >= 1.0:
        return cp_incompressible
        
    m2 = mach ** 2
    beta = np.sqrt(1 - m2)
    
    # Standard Kármán-Tsien formula
    cp_compressible = cp_incompressible / (beta + (m2 / (1 + beta)) * (cp_incompressible / 2.0))
    
    return cp_compressible

def apply_3d_corrections(cl_2d: float, cm_2d: float, alpha_deg: float, 
                         aspect_ratio: float, mach: float = 0.0) -> tuple:
    """
    Approximates finite 3D wing effects based on classical lifting-line theory.
    Returns the corrected (Cl_3d, Cd_induced, Cm_3d).
    """
    if aspect_ratio <= 0.1:  # Infinite wing (2D)
        return cl_2d, 0.0, cm_2d
        
    beta = np.sqrt(max(0.001, 1 - mach**2))
    
    # Lift correction for finite span (Anderson / JavaFoil formula)
    # The lift curve slope is reduced by the downwash of the tip vortices.
    cl_3d = cl_2d / (beta + (2.0 / aspect_ratio))
    
    # Induced drag (vortex drag)
    # Assumes an elliptical lift distribution (k=1.0)
    cd_induced = (cl_3d ** 2) / (np.pi * aspect_ratio)
    
    # Pitching moment reduction due to pressure loss at wingtips
    cm_3d = cm_2d * (aspect_ratio / (aspect_ratio + 4.0))
    
    return cl_3d, cd_induced, cm_3d
