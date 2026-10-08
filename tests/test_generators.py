"""Trailing-edge option of the NACA generators (DECISIONS.md, M1 D1)."""

import numpy as np
import pytest

from foil_lab.generators import generate_naca4, generate_naca5


def _old_closed_naca4(code: str, n: int) -> np.ndarray:
    """The v1.3.1 closed-trailing-edge formula, copied verbatim, to pin the default."""
    m, p, t = int(code[0]) / 100.0, int(code[1]) / 10.0, int(code[2:4]) / 100.0
    x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n // 2 + 1)))
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)
    yc, dy = np.zeros_like(x), np.zeros_like(x)
    if m > 0:
        f, b = x <= p, x > p
        yc[f] = (m / p**2) * (2 * p * x[f] - x[f]**2)
        dy[f] = (2 * m / p**2) * (p - x[f])
        yc[b] = (m / (1 - p)**2) * ((1 - 2 * p) + 2 * p * x[b] - x[b]**2)
        dy[b] = (2 * m / (1 - p)**2) * (p - x[b])
    th = np.arctan(dy)
    up = np.column_stack(((x - yt * np.sin(th))[::-1], (yc + yt * np.cos(th))[::-1]))
    lo = np.column_stack(((x + yt * np.sin(th))[1:], (yc - yt * np.cos(th))[1:]))
    return np.vstack((up, lo))


@pytest.mark.parametrize("code", ["0012", "2412", "4412"])
def test_default_is_bit_identical_to_v131(code):
    assert np.array_equal(generate_naca4(code, 160), _old_closed_naca4(code, 160))
    assert np.array_equal(generate_naca4(code, 160, trailing_edge="closed"),
                          generate_naca4(code, 160))


@pytest.mark.parametrize("gen,code", [(generate_naca4, "2412"), (generate_naca5, "23012")])
def test_trailing_edge_gap(gen, code):
    closed = gen(code, 160)
    opened = gen(code, 160, trailing_edge="open")
    assert np.hypot(*(closed[0] - closed[-1])) < 1e-15
    # gap = 2 * y_t(1) = 10 t (0.2969 - 0.1260 - 0.3516 + 0.2843 - 0.1015) = 0.00252 for t = 0.12
    assert np.hypot(*(opened[0] - opened[-1])) == pytest.approx(0.00252, abs=1e-9)


def test_invalid_trailing_edge_rejected():
    with pytest.raises(ValueError, match="trailing_edge"):
        generate_naca4("2412", 160, trailing_edge="blunt")
