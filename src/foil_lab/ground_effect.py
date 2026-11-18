"""Ground effect via the method of images (rigid wall at y = 0), on the inviscid core.

The configuration is pitched by alpha about a pivot (default the moment reference point), raised
so that the pivot is at the given height and solved together with its mirror image about y = 0 in
a stream parallel to the ground. By symmetry the mirrored pair has y = 0 as a streamline, so it is
the flow over the section above a rigid wall. All bodies go through the one assembly of
:mod:`foil_lab.inviscid` (each with its own Kutta condition), so the image is not a separate
kernel. Loads are those of the real bodies, in the ground (wind) axes.

Behaviour change from v1.3.1: there the stream was inclined by alpha to an unrotated section,
so it crossed the wall; here the section is pitched and the stream is parallel to the ground
(DECISIONS.md, M2 step 3). Free surface (high Froude) is out of scope.

:func:`ground_frame` and :func:`image` are the one implementation of this construction, used by
``fl.analyze`` (``ground_height``, ``pivot``; any number of elements, inviscid or viscous) and
:func:`foil_lab.flowfield.velocity_field`. Validation: tests/test_corrections.py solves the same
pair of bodies as a free-air two-element configuration and finds the same loads, and checks that
the velocity normal to the ground vanishes on y = 0 (SPEC.md 4.2).
"""

from __future__ import annotations

import numpy as np

from .geometry import rotate


def ground_frame(alpha_deg: float, pivot, height: float):
    """Point map into the ground frame (ground y = 0): pitch by ``alpha_deg`` (nose up positive)
    about ``pivot``, then move up so that the pivot is ``height`` above the ground."""
    px, py = float(pivot[0]), float(pivot[1])

    def fn(p):
        out = rotate(np.asarray(p, dtype=float), alpha_deg, pivot=(px, py))
        out[:, 1] += height - py
        return out
    return fn


def image(coords) -> np.ndarray:
    """Mirror image of a body about the ground y = 0 (the panel set accepts either orientation)."""
    return np.asarray(coords, dtype=float) * np.array([1.0, -1.0])
