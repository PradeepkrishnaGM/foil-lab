"""Foil-Lab: 2D panel-method analysis of single and multi-element airfoil sections.

Public interface (inviscid in this version; viscous analysis follows in M3)::

    import foil_lab as fl

    af = fl.Airfoil.naca("2412")
    res = fl.analyze(af, alpha=4.0)
    pol = fl.polar(af, alphas=range(-4, 15))

The submodules (``geometry``, ``inviscid``, ``solver``, ``generators``, ...) stay importable.
"""

__version__ = "2.0.0.dev0"

from .airfoil import Airfoil, Configuration, Element  # noqa: E402
from .api import ElementResult, Polar, Result, analyze, polar  # noqa: E402

__all__ = ["Airfoil", "Element", "Configuration", "analyze", "polar", "Result", "ElementResult",
           "Polar", "__version__"]
