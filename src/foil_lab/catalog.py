"""Airfoil import by name (SPEC.md 4.4).

``Airfoil.from_name("naca2412")`` uses Foil-Lab's own NACA generators. Any other name is looked up
in the UIUC Airfoil Coordinates Database (https://m-selig.ae.illinois.edu/ads/coord_database.html)
and downloaded on demand into a cache outside the repository: ``$FOIL_LAB_CACHE/airfoils`` or
``~/.cache/foil-lab/airfoils``. No coordinate data is bundled with Foil-Lab; the airfoil's
``source`` records the URL and the SHA-256 of the file that was read.
"""

from __future__ import annotations

import hashlib
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

UIUC_URL = "https://m-selig.ae.illinois.edu/ads/coord/{name}.dat"
_NAME = re.compile(r"^[a-z0-9][a-z0-9_.\-]*$")
_NACA = re.compile(r"^naca\s*[-_ ]?\s*(\d{4,5})$")


class AirfoilUnavailable(RuntimeError):
    """A named airfoil is neither cached nor downloadable (unknown name, or no network)."""


def cache_dir() -> Path:
    """Download cache: ``$FOIL_LAB_CACHE`` or ``~/.cache/foil-lab``, plus ``airfoils/``."""
    root = os.environ.get("FOIL_LAB_CACHE") or Path.home() / ".cache" / "foil-lab"
    return Path(root) / "airfoils"


def naca_code(name: str) -> Optional[str]:
    """The digits of a NACA 4- or 5-digit name ("NACA 2412", "naca23012"), else ``None``."""
    m = _NACA.match(name.strip().lower())
    return m.group(1) if m else None


def fetch_uiuc(name: str, timeout_s: float = 30.0, sha256: Optional[str] = None) -> tuple[bytes, str, str]:
    """``(file bytes, url, sha256)`` of a UIUC coordinate file, from the cache if present.

    ``name`` is the file name in the database without ``.dat`` (lower case, e.g. ``"e387"``,
    ``"clarky"``). With ``sha256`` the file must have that checksum (a changed upstream or cache file
    is an error, not silently used).

    Raises
    ------
    ValueError
        For a name with characters other than letters, digits, ``_``, ``-`` and ``.``, or a checksum
        mismatch.
    AirfoilUnavailable
        If the file is not cached and cannot be downloaded.
    """
    key = name.strip().lower()
    if key.endswith(".dat"):
        key = key[:-4]
    if not _NAME.match(key):
        raise ValueError(f"not a valid airfoil name: {name!r}")
    url = UIUC_URL.format(name=key)
    path = cache_dir() / f"{key}.dat"
    if path.exists():
        data = path.read_bytes()
    else:
        try:
            with urllib.request.urlopen(url, timeout=timeout_s) as resp:
                data = resp.read()
        except (urllib.error.URLError, OSError) as exc:
            raise AirfoilUnavailable(f"{name}: not cached and the download from {url} failed ({exc})") from exc
        if b"<html" in data[:512].lower():
            raise AirfoilUnavailable(f"{name}: {url} did not return a coordinate file")
        if sha256 is not None and hashlib.sha256(data).hexdigest() != sha256:
            raise ValueError(f"{name}: checksum {hashlib.sha256(data).hexdigest()} does not match {sha256}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    if sha256 is not None and digest != sha256:
        raise ValueError(f"{name}: cached file checksum {digest} does not match {sha256}")
    return data, url, digest
