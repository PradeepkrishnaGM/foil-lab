from foil_lab.utils.io import load_dat
from foil_lab.geometry import normalize_airfoil, repanel_cosine
from foil_lab.analysis import run_alpha_sweep, find_zero_lift_angle, calculate_aerodynamic_center
from foil_lab.plotting import plot_geometry, plot_cp, plot_polar

def main():
    # 1. Provide a path to a standard airfoil .dat file (e.g., NACA 2412)
    # Ensure you have a 'naca2412.dat' file in this directory to run the test
    filepath = "naca2412.dat" 
    
    print(f"Loading {filepath}...")
    try:
        raw_coords = load_dat(filepath)
    except FileNotFoundError:
        print(f"Error: Please download a '{filepath}' file to run this example.")
        return

    # 2. Geometry Pipeline
    print("Normalizing and repaneling geometry...")
    normalized_coords = normalize_airfoil(raw_coords)
    panels = repanel_cosine(normalized_coords, num_panels=120)
    
    # 3. Plot the smoothed geometry
    plot_geometry(panels, title="NACA 2412 (120 Cosine Panels)", show_panels=True)

    # 4. Run an Alpha Sweep (-5 to 10 degrees)
    print("Running inviscid solver alpha sweep (-5° to 10°)...")
    polar = run_alpha_sweep(panels, alpha_start=-5.0, alpha_end=10.0, alpha_step=1.0)
    
    # 5. Extract Engineering Metrics
    alpha_L0 = find_zero_lift_angle(polar)
    x_ac, _ = calculate_aerodynamic_center(polar)
    
    print("\n--- Results ---")
    print(f"Zero-Lift Angle (alpha_L0): {alpha_L0:.2f} degrees")
    print(f"Aerodynamic Center (x_ac):  {x_ac:.3f} c")
    
    # 6. Visualizations
    plot_polar(polar)
    
    # Plot Cp for a specific angle of attack (e.g., alpha = 5 degrees)
    # The sweep goes from -5 to 10 by 1, so index 10 corresponds to 5 degrees
    target_idx = 10 
    sol_5deg = polar.solutions[target_idx]
    plot_cp(panels, sol_5deg)

if __name__ == "__main__":
    main()
