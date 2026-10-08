import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from .solver import SolutionData, compute_panel_geometry
from .analysis import PolarData

def plot_geometry(coords: np.ndarray, title: str = "Airfoil Geometry", show_panels: bool = False):
    """
    Plots the normalized airfoil geometry. 
    Optionally displays the panel nodes to verify repaneling quality.
    """
    plt.figure(figsize=(10, 4))
    
    if show_panels:
        plt.plot(coords[:, 0], coords[:, 1], 'o-', markersize=3, label='Panel Nodes')
    else:
        plt.plot(coords[:, 0], coords[:, 1], '-', linewidth=1.5, label='Surface')
        
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.title(title)
    plt.xlabel("x/c")
    plt.ylabel("y/c")
    plt.axis("equal")  # Crucial for airfoils: maintains true thickness ratio
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_cp(coords: np.ndarray, solution: SolutionData, title: Optional[str] = None):
    """
    Plots the pressure coefficient (Cp) distribution over the airfoil.
    In aerospace convention, the Cp axis is inverted so that the 
    suction peak (negative Cp) is at the top of the graph.
    """
    geom = compute_panel_geometry(coords)
    mid_x = geom[4]
    cp = solution.cp_dist
    
    # Split upper and lower surfaces for cleaner line plotting
    # The panel array wraps around from TE-Upper to LE to TE-Lower
    le_panel_idx = np.argmin(mid_x)
    
    x_upper = mid_x[:le_panel_idx+1]
    cp_upper = cp[:le_panel_idx+1]
    
    x_lower = mid_x[le_panel_idx:]
    cp_lower = cp[le_panel_idx:]
    
    plt.figure(figsize=(8, 5))
    plt.plot(x_upper, cp_upper, 'b-', label='Upper Surface')
    plt.plot(x_lower, cp_lower, 'r-', label='Lower Surface')
    
    # Invert Y axis: Negative Cp (suction) goes UP
    plt.gca().invert_yaxis()
    
    plot_title = title if title else f"Cp Distribution ($\\alpha$ = {solution.alpha_deg:.1f}°)"
    plt.title(plot_title)
    plt.xlabel("x/c")
    plt.ylabel("$C_p$")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_polar(polar: PolarData):
    """
    Generates a multi-plot figure showing Cl vs Alpha and Cm vs Alpha.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Cl vs Alpha
    ax1.plot(polar.alphas, polar.cls, 'o-', color='navy', markersize=4)
    ax1.set_title("Lift Coefficient vs. Angle of Attack")
    ax1.set_xlabel("$\\alpha$ (degrees)")
    ax1.set_ylabel("$C_l$")
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.axvline(0, color='black', linewidth=0.5)
    ax1.grid(True, linestyle=':', alpha=0.7)
    
    # Cm vs Alpha (Quarter-chord moment)
    ax2.plot(polar.alphas, polar.cm_qcs, 's-', color='darkred', markersize=4)
    ax2.set_title("Pitching Moment ($C_{m, c/4}$) vs. Angle of Attack")
    ax2.set_xlabel("$\\alpha$ (degrees)")
    ax2.set_ylabel("$C_{m, c/4}$")
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.axvline(0, color='black', linewidth=0.5)
    ax2.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
