"""Foil-Lab: 2D panel-method analysis of single and multi-element airfoil sections.

Public interface::

    import foil_lab as fl

    af = fl.Airfoil.naca("2412")
    res = fl.analyze(af, alpha=4.0)                       # inviscid
    res = fl.analyze(af, alpha=4.0, re=3e6)               # coupled viscous-inviscid
    pol = fl.polar(af, alphas=range(-4, 15), re=3e6)
    df = pol.to_dataframe()                               # pandas (optional dependency)
    flapped = af.with_flap(hinge_x=0.75, deflection=10.0)
    res = fl.analyze(af, 4.0, mach=0.5, aspect_ratio=8.0) # corrections, recorded in res.corrections
    e387 = fl.Airfoil.from_name("e387")                   # UIUC database, cached on first use

    from foil_lab import plot, export                     # figures (fig, axes) and file exports
    db = fl.PolarDatabase("polars.sqlite")                # stored polars, reused by geometry hash

Lower-level modules (``inviscid``, ``viscous``, ``geometry``, ``corrections``, ...) stay importable.
"""

__version__ = "2.0.0"

from .airfoil import Airfoil, Configuration, Element  # noqa: E402
from .api import ElementResult, Polar, Result, analyze, polar  # noqa: E402
from .flap_polars import FlapPolarSet, FlapSlice, flap_polars  # noqa: E402
from .database import PolarDatabase, geometry_hash  # noqa: E402

__all__ = ["Airfoil", "Element", "Configuration", "analyze", "polar", "Result", "ElementResult",
           "Polar", "flap_polars", "FlapPolarSet", "FlapSlice", "PolarDatabase", "geometry_hash", "__version__"]
