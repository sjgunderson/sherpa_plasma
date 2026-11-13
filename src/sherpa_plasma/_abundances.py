"""Numerical abundance compositions shared by model capture and atomic queries.

Compositions are linear number abundances relative to hydrogen; fitted element
multipliers remain separate. XSPEC is imported only when resolving its tables.
"""

from collections.abc import Mapping
from pathlib import Path

import numpy as np


def _abundance_values(values):
    from .plasma import _SYMBOLS, _element

    result = {symbol: 0. for symbol in _SYMBOLS[1:]}
    if isinstance(values, Mapping):
        seen = set()
        for element, value in values.items():
            z = _element(element)
            if z in seen:
                raise ValueError("duplicate element in abundance composition")
            seen.add(z)
            number = np.asarray(value, dtype=float)
            if number.ndim != 0 or not np.isfinite(number) or number < 0:
                raise ValueError("abundance values must be finite nonnegative scalars")
            result[_SYMBOLS[z]] = float(number)
        return result
    array = np.asarray(values, dtype=float)
    if array.shape == (31,):
        array = array[1:]
    if array.shape != (30,) or not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError("abundance vector must contain 30 finite nonnegative values (H through Zn)")
    return dict(zip(_SYMBOLS[1:], map(float, array)))


def _xspec_abundance_values(table=None, *, allow_missing=False):
    try:
        from sherpa.astro import xspec
    except ImportError as exc:
        if allow_missing:
            return None
        raise ImportError(
            "resolving an XSPEC abundance table requires Sherpa/XSPEC; "
            "provide numerical abundances or a 30-value abundance file for offline use"
        ) from exc
    try:
        values = xspec.get_xsabundances(table)
    except ValueError:
        if allow_missing:
            return None
        raise
    return _abundance_values(values)


def _resolve_abundance_spec(spec, *, xspec=False):
    """Return a detached composition, native marker, or ``atomdb:`` name.

    ``xspec=True`` is used for legacy named captures: it must not reinterpret
    an XSPEC name as a different local AtomDB table. Normal query names first
    try XSPEC, then fall back to an explicitly available local AtomDB name.
    """
    if spec is None:
        if xspec:
            raise ValueError("captured component requires abundance_values or an XSPEC table name")
        return "native"
    if not isinstance(spec, (str, Path)):
        return _abundance_values(spec)
    name = str(spec).strip()
    if not name:
        raise ValueError("provide a nonempty abundance-table name")
    if name.lower().startswith("atomdb:"):
        local = name.split(":", 1)[1].strip()
        if not local:
            raise ValueError("provide an AtomDB abundance-table name after 'atomdb:'")
        return f"atomdb:{local}"
    if name.lower().startswith("xspec:"):
        return _xspec_abundance_values(name.split(":", 1)[1])
    if not xspec and name.lower() == "native":
        return "native"
    path = Path(name).expanduser()
    if path.is_file():
        return _abundance_values(np.loadtxt(path, dtype=float))
    if isinstance(spec, Path) or "/" in name or "\\" in name:
        raise FileNotFoundError(f"abundance file does not exist: {path}")
    if xspec:
        if name.split(":", 1)[0].lower() == "file":
            raise ValueError(
                "a captured custom table requires abundance_values; recapture "
                "the model or supply its numerical composition"
            )
        return _xspec_abundance_values(name.split(":", 1)[0])
    values = _xspec_abundance_values(name, allow_missing=True)
    return f"atomdb:{name}" if values is None else values
