"""Named closure sets of the boundary layer, selected by a parameter.

A closure set is a module exposing the same functions and constants (``INTERFACE``); the
equations, discretisation and Newton solver exist once and only call these. Formulas are never
mixed between sets.

* ``"fidkowski2022"`` (:mod:`foil_lab.bl.fidkowski2022`): Fidkowski's published closures, close to
  later XFOIL versions; the SPEC 4.1 targets are judged with this set.
* ``"drela1987"`` (:mod:`foil_lab.bl.drela1987`): Drela & Giles (1987) with Drela (1989).

Each module's ``PROVENANCE`` names the source and equation of every closure. Why there are two
sets: DECISIONS.md, M3, "Two closure sets".
"""

from __future__ import annotations

from types import ModuleType

from . import drela1987, fidkowski2022

#: functions and constants every closure set provides (signatures in the set modules)
INTERFACE = ("NAME", "PROVENANCE", "ETA_D", "hk", "hstar_lam", "cf_lam", "cd_lam", "hstar_turb",
             "cf_turb", "us", "delta", "ctau_eq", "cd_turb", "lag_k", "uq", "dn_dxi",
             "ctau_transition", "ETA_D_WAKE", "WAKE_DTDX_CLIP", "hk_wake", "us_wake", "ctau_eq_wake",
             "cd_wake", "uq_wake", "ctau_wake_start")

CLOSURE_SETS: dict[str, ModuleType] = {m.NAME: m for m in (fidkowski2022, drela1987)}


def get(name_or_set) -> ModuleType:
    """The closure set of the given name (or the set itself)."""
    if isinstance(name_or_set, ModuleType):
        return name_or_set
    try:
        return CLOSURE_SETS[name_or_set]
    except KeyError:
        raise ValueError(f"unknown closure set {name_or_set!r}; known: {sorted(CLOSURE_SETS)}") from None


for _m in CLOSURE_SETS.values():
    _missing = [a for a in INTERFACE if not hasattr(_m, a)]
    if _missing:
        raise ImportError(f"closure set {_m.NAME} lacks {_missing}")
