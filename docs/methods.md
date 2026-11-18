# Methods and sources

Each method is implemented from a published source, cited in the code where it is used.
"Checked" means the formula was taken from the source with the source at hand; "unverified"
means the citation was written without the book or paper at hand and its section number or
detail still needs checking against the source.

| Part | Method | Source | Status |
|---|---|---|---|
| Inviscid core (`inviscid.py`) | Linear-strength vortex panels, one closure condition per body, trailing-edge panel for an open trailing edge, Kutta condition | M. Drela, "XFOIL: An Analysis and Design System for Low Reynolds Number Airfoils", Low Reynolds Number Aerodynamics, Springer, 1989 (LNE89), eqs. 1-9 | checked |
| | Panel influence formulas | J. Katz and A. Plotkin, *Low-Speed Aerodynamics*, 2nd ed., CUP, 2001, sections 10.2-10.4, 11.4.2 | unverified (sections) |
| | Vortex panel method | J. L. Kuethe and C.-Y. Chow, *Foundations of Aerodynamics*, 5th ed., Wiley, 1998, section 5.10 | unverified (section) |
| Boundary layer and coupling (`viscous.py`, `bl/`) | Integral boundary layer, lag-entrainment turbulent model, e^n envelope transition, one Newton system with the edge-velocity relation | M. Drela and M. B. Giles, "Viscous-Inviscid Analysis of Transonic and Low Reynolds Number Airfoils", AIAA J. 25(10), 1987 (DG87); LNE89; M. Drela, AIAA-89-2200, 1989 (BTE89) | checked (DG87 eq. 18 reading to verify) |
| | Closure set "fidkowski2022" (the default), wake and dead-air treatment, Newton limits | K. J. Fidkowski, "A Coupled Inviscid-Viscous Airfoil Analysis Solver, Revisited" (FID), appendix B and Table 2 | checked (from the page images) |
| | Profile drag at the end of the wake | H. B. Squire and A. D. Young, "The Calculation of the Profile Drag of Aerofoils", ARC R&M 1838, 1937 | checked through FID / LNE89 |
| Displacement effect | Transpiration (source) model | M. J. Lighthill, "On displacement thickness", J. Fluid Mech. 4, 1958 | unverified |
| Geometry (`generators.py`) | NACA 4- and 5-digit sections, open trailing-edge coefficient -0.1015 | E. N. Jacobs, K. E. Ward, R. M. Pinkerton, NACA Report 460, 1933; I. H. Abbott and A. E. von Doenhoff, *Theory of Wing Sections*, 1959 | unverified |
| Interpolation | PCHIP in flap deflection | F. N. Fritsch and R. E. Carlson, SIAM J. Numer. Anal. 17, 1980 | standard (SciPy) |
| Hinge moment sign | Trailing edge down positive | Abbott and von Doenhoff (1959) | unverified (page not found) |
| Exact solutions (tests) | Kármán-Trefftz airfoil | L. M. Milne-Thomson, *Theoretical Aerodynamics*, 1958, ch. 7; Katz and Plotkin (2001) section 6.7 | unverified (sections) |
| | Two-element exact potential flow | B. R. Williams, ARC R&M 3717, 1973 | checked |
| | Thin-airfoil flap slopes | H. Glauert, as given in J. D. Anderson, *Fundamentals of Aerodynamics* | unverified (section) |
| Compressibility (`corrections.py`) | Kármán-Tsien rule; isentropic critical pressure coefficient | Th. von Kármán, J. Aero. Sci. 8, 1941; H. S. Tsien, J. Aero. Sci. 6, 1939 | unverified |
| Sweep | Simple sweep theory (independence principle) | A. Busemann, 1935; H. Schlichting and E. Truckenbrodt, *Aerodynamics of the Airplane*, 1979 | unverified |
| Finite wing | Lift factor of the elliptic wing; induced drag | H. B. Helmbold, Jahrbuch der deutschen Luftfahrtforschung, 1942; L. Prandtl's lifting-line theory | unverified |
| Ground effect | Method of images (rigid wall) | standard potential-flow result; validated in the tests by the free-air mirror problem | tested |

The viscous model follows the published formulation of Drela & Giles (1987) and Drela (1989); the
default closure set "fidkowski2022" follows Fidkowski's later published forms (close to later
XFOIL versions). The older set "drela1987" is kept but not benchmarked (DECISIONS.md, M3).
