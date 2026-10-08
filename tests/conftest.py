"""Shared pytest configuration.

Tests marked ``xfoil`` run the external XFOIL program and are skipped when either
``xfoil`` or ``xvfb-run`` is not on the PATH (XFOIL 6.99 as packaged for Fedora
must run with graphics enabled, so it is started under a virtual X server).
"""

from __future__ import annotations

import shutil

import pytest

_XFOIL_MISSING = [name for name in ("xfoil", "xvfb-run") if shutil.which(name) is None]


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip ``xfoil``-marked tests when the external tools are unavailable."""
    if not _XFOIL_MISSING:
        return
    skip = pytest.mark.skip(reason=f"not found on PATH: {', '.join(_XFOIL_MISSING)}")
    for item in items:
        if "xfoil" in item.keywords:
            item.add_marker(skip)
