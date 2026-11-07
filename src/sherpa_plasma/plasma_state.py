"""Read-only snapshots of existing Sherpa/XSPEC APEC components.

No model evaluation, fitting component, or atomic database is defined here.
"""
from dataclasses import dataclass
from types import MappingProxyType
from collections.abc import Mapping
import math

from .plasma import _SYMBOLS

_APEC_TYPES = frozenset(("xsapec", "xsbapec", "xsvapec", "xsbvapec",
                         "xsvvapec", "xsbvvapec"))


@dataclass(frozen=True)
class PlasmaState:
    """Detached parameter snapshot; temperature is keV, velocity is km/s.

    ``parameters`` preserves the component's own parameter names. Abundances
    remain relative to XSPEC's abundance table; they are not converted to the
    local AtomDB abundance set. Compound-expression scale factors and
    absorption are not applied to the component normalization.
    """

    name: str
    model_type: str
    parameters: Mapping[str, float]
    abundance_table: str | None = None
    xset: Mapping[str, str] | None = None

    def __post_init__(self):
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))
        object.__setattr__(self, "xset", MappingProxyType(dict(self.xset or {})))

    def _value(self, name, default=None):
        return next((v for k, v in self.parameters.items() if k.lower() == name), default)

    @property
    def kT(self):
        return self._value("kt")

    @property
    def redshift(self):
        return self._value("redshift", 0.)

    @property
    def norm(self):
        return self._value("norm")

    @property
    def velocity(self):
        return self._value("velocity", 0.)

    @property
    def metal_abundance(self):
        """The scalar Abundanc parameter of apec/bapec, when present."""
        return self._value("abundanc")

    @property
    def abundances(self):
        """Explicit elemental parameters only; no inferred trace abundances."""
        return {k: v for k, v in self.parameters.items() if k in _SYMBOLS[1:]}


def read_plasma_model(model=None, *, dataset_id=None, session=None):
    """Read the current APEC components, including resolved linked values.

    Supply a source expression, or omit it to read ``session.get_source``.
    The default session is ``sherpa.astro.ui``. Shared-grid wrappers and
    compound expressions are traversed through their ``parts``. Each unique
    component is reported once, in expression order. Other components are
    ignored; an expression without supported APEC components raises ValueError.

    Supports XSapec, XSbapec, XSvapec, XSbvapec, XSvvapec and XSbvvapec.
    XSPEC global settings are captured for real XSPEC components, but are
    never changed. Reread after a fit or parameter/settings change.
    """
    if model is not None and (dataset_id is not None or session is not None):
        raise ValueError("supply a model or a dataset/session, not both")
    if model is None:
        if session is None:
            from sherpa.astro import ui as session
        model = session.get_source() if dataset_id is None else session.get_source(dataset_id)

    found, visited = [], set()

    def visit(part):
        if id(part) in visited:
            return
        visited.add(id(part))
        model_type = type(part).__name__.lower()
        if model_type in _APEC_TYPES:
            found.append((part, model_type))
            return
        for child in getattr(part, "parts", ()):
            visit(child)

    visit(model)
    if not found:
        raise ValueError("source contains no supported built-in APEC components")

    abundance_table, xset = None, {}
    if any(type(part).__module__.startswith("sherpa.astro.xspec") for part, _ in found):
        from sherpa.astro import xspec
        abundance_table, xset = xspec.get_xsabund(), xspec.get_xsxset()

    result = []
    for part, model_type in found:
        parameters = {p.name: float(p.val) for p in part.pars}
        if not all(math.isfinite(v) for v in parameters.values()):
            raise ValueError(f"nonfinite parameter in {part.name}")
        if not any(k.lower() == "kt" for k in parameters):
            raise ValueError(f"missing kT parameter in {part.name}")
        state = PlasmaState(part.name, model_type, parameters, abundance_table, xset)
        if state.kT <= 0 or state.redshift <= -1:
            raise ValueError(f"invalid temperature or redshift in {part.name}")
        result.append(state)
    return tuple(result)
