import numpy as np
from pathlib import Path

def load_dat(filepath: str | Path) -> np.ndarray:
    """
    Loads an airfoil .dat file and returns a numpy array of coordinates.
    Expects standard Selig format (TE Upper -> LE -> TE Lower).
    Automatically skips text headers.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Airfoil file not found: {filepath}")
        
    coords = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    # Attempt to parse the first two columns as floats (X, Y)
                    x, y = float(parts[0]), float(parts[1])
                    coords.append([x, y])
                except ValueError:
                    # If conversion fails, it's likely a header string; skip it
                    continue
                    
    return np.array(coords)

def save_dat(filepath: str | Path, coords: np.ndarray, header: str = "Foil Lab Airfoil"):
    """
    Saves airfoil coordinates to a .dat file.
    """
    with open(filepath, 'w') as f:
        f.write(f"{header}\n")
        for x, y in coords:
            f.write(f"{x:.6f} {y:.6f}\n")
            
# Add to the bottom of src/foil_lab/utils/io.py
import csv
from ..analysis import PolarData # Note the relative import 

def export_polar_csv(filepath: str | Path, polar: PolarData):
    """Saves the macroscopic polar coefficients to a standard CSV file."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Alpha (deg)", "Cl", "Cm_LE", "Cm_c/4"])
        
        for i in range(len(polar.alphas)):
            writer.writerow([
                f"{polar.alphas[i]:.3f}", 
                f"{polar.cls[i]:.4f}", 
                f"{polar.cm_les[i]:.4f}", 
                f"{polar.cm_qcs[i]:.4f}"
            ])
