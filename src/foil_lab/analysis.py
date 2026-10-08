import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from .solver import solve_flow, SolutionData

@dataclass
class PolarData:
    """
    Structured container for an alpha sweep. 
    Aggregates macroscopic coefficients for quick polar plotting,
    while retaining the full SolutionData for local Cp analysis.
    """
    alphas: np.ndarray
    cls: np.ndarray
    cds: np.ndarray    # NEW: Store Induced Drag
    cm_les: np.ndarray
    cm_qcs: np.ndarray
    solutions: List[SolutionData]

def run_alpha_sweep(coords: np.ndarray, alpha_start: float, alpha_end: float, 
                    alpha_step: float = 1.0, v_inf: float = 1.0,
                    mach: float = 0.0, reynolds: float = 0.0, aspect_ratio: float = 0.0,
                    roughness: int = 0, enable_post_stall: bool = False) -> PolarData:
    """
    Executes the aerodynamic solver across a specified range of angles of attack.

    Args:
        coords: Normalized, repaneled airfoil coordinates.
        alpha_start: Starting angle of attack in degrees.
        alpha_end: Ending angle of attack in degrees.
        alpha_step: Increment between angles in degrees.
        v_inf: Freestream velocity.
        roughness: Surface roughness state 0-3 (viscous transition/drag).
        enable_post_stall: Apply empirical post-stall Cl/Cd corrections.

    Returns:
        PolarData: A dataclass containing arrays of coefficients and full solution data.
    """
    # Calculate the number of points to ensure we hit the alpha_end exactly
    num_steps = int(np.round((alpha_end - alpha_start) / alpha_step)) + 1
    alphas = np.linspace(alpha_start, alpha_end, num_steps)
    
    cls, cds, cm_les, cm_qcs, solutions = [], [], [], [], [] # Add cds list
    
    for alpha in alphas:
        sol = solve_flow(coords, alpha, v_inf, mach, reynolds, aspect_ratio,
                         roughness=roughness, enable_post_stall=enable_post_stall)
        
        cls.append(sol.cl)
        # Track the TOTAL drag (Viscous + Induced) for the drag polar
        cds.append(sol.cd_total) 
        cm_les.append(sol.cm_le)
        cm_qcs.append(sol.cm_qc)
        solutions.append(sol)
        
    return PolarData(
        alphas=np.array(alphas),
        cls=np.array(cls),
        cds=np.array(cds), # Save it to the dataclass
        cm_les=np.array(cm_les),
        cm_qcs=np.array(cm_qcs),
        solutions=solutions
    )

def find_zero_lift_angle(polar: PolarData) -> float:
    """
    Interpolates the alpha sweep data to find the exact angle of attack 
    where the lift coefficient (Cl) is zero.
    """
    # Use numpy.interp to find where Cl crosses 0. 
    # Note: np.interp requires the x-axis (Cl) to be monotonically increasing.
    # For standard attached flow polars, Cl increases with alpha.
    return np.interp(0.0, polar.cls, polar.alphas)

def calculate_aerodynamic_center(polar: PolarData) -> Tuple[float, float]:
    """
    Estimates the location of the aerodynamic center (AC) based on the 
    change in pitching moment with respect to lift (dCm / dCl).
    Returns the X-coordinate of the AC (normalized chord fraction).
    """
    # Linear fit of Cm_le vs Cl to find the slope
    # Cm_le = Cm_ac - x_ac * Cl  =>  dCm_le / dCl = -x_ac
    slope, _ = np.polyfit(polar.cls, polar.cm_les, 1)
    
    # The AC is located at -slope (usually around 0.25 for thin airfoils)
    x_ac = -slope
    
    # Inviscid theory assumes AC is on the chord line (y=0)
    y_ac = 0.0 
    
    return x_ac, y_ac
