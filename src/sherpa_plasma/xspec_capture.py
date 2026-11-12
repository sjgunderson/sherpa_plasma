"""Capture XSPEC metadata for CIE and AtomDB-compatible NEI plasma components.

The captured arrays are raw XSPEC metadata. This module does not assign
physical units, rescale them by model normalization, or calculate line fluxes.
Single-temperature models that do not publish arrays use their line
temperature with unit weight; multi-temperature models require published
temperature/weight arrays. Supported NEI and plane-parallel shock models use
their explicit parameters rather than CIE temperature/weight arrays.
"""
from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType
from ._abundances import _abundance_values, _xspec_abundance_values

import numpy as np


_CIE_TYPES = frozenset((
    "xsapec", "xsbapec", "xsbvapec", "xsbvvapec", "xsvapec", "xsvvapec",
    "xsrsapec", "xsrsvapec", "xsrsvvapec", "xssnapec", "xsbsnapec",
    "xstapec", "xsbtapec", "xsvtapec", "xsbvtapec", "xsvvtapec",
    "xsbvvtapec", "xscie", "xsbcie", "xsbvcie", "xsbvvcie", "xsvcie",
    "xsvvcie", "xsequil", "xsbequil", "xsbvequil", "xsvequil", "xscoolflow",
    "xsbcoolflow", "xsbvcoolflow", "xsvcoolflow", "xscflow", "xsmkcflow",
    "xsvmcflow", "xscemekl", "xscevmkl", "xscempow", "xsbcempow",
    "xsvcempow", "xsbvcempow",
    "xsgadem", "xsbgadem", "xsvgadem", "xsbvgadem", "xsvvgadem",
    "xsbvvgadem", "xswdem", "xsbwdem", "xsvwdem", "xsbvwdem",
    "xsvvwdem", "xsbvvwdem", "xscheb6", "xsbcheb6", "xsvcheb6",
    "xsbvcheb6", "xsexpcheb6", "xsbexpcheb6", "xsvexpcheb6",
    "xsbvexpcheb6", "xsmekal", "xsvmekal", "xsraymond", "xsvraymond",
    "xscph", "xsbcph", "xsbvcph", "xsvcph",
))
_NEI_INITIAL_POPULATIONS = {
    model_type: "ionizing"
    for model_type in (
        "xsnei", "xsvnei", "xsvvnei",
    )
}
_NEI_INITIAL_POPULATIONS.update({
    model_type: "initial_temperature"
    for model_type in (
        "xsgnei", "xsvgnei", "xsvvgnei", "xsrnei", "xsvrnei", "xsvvrnei",
    )
})
_PSHOCK_TYPES = frozenset((
    "xspshock", "xsvpshock", "xsvvpshock",
))
_NEI_TYPES = frozenset(_NEI_INITIAL_POPULATIONS)
_SUPPORTED_TYPES = _CIE_TYPES | _NEI_TYPES | _PSHOCK_TYPES
_SINGLE_TEMPERATURE_PARAMETERS = {
    model_type: "kT"
    for model_type in (
        "xsapec", "xsbapec", "xsbvapec", "xsbvvapec", "xsvapec",
        "xsvvapec", "xsrsapec", "xsrsvapec", "xsrsvvapec", "xssnapec",
        "xsbsnapec", "xscie", "xsbcie", "xsbvcie", "xsbvvcie",
        "xsvcie", "xsvvcie", "xsbequil", "xsbvequil", "xsvequil",
        "xsequil", "xsmekal", "xsvmekal", "xsraymond", "xsvraymond",
    )
}
_SINGLE_TEMPERATURE_PARAMETERS.update({
    model_type: "kTi"
    for model_type in (
        "xstapec", "xsbtapec", "xsvtapec", "xsbvtapec", "xsvvtapec",
        "xsbvvtapec",
    )
})
_CAPTURE_LOCK = RLock()


@dataclass(frozen=True)
class CapturedPlasmaComponent:
    """Detached parameters and optional raw XSPEC temperature/DEM arrays.

    NEI and PShock components use their captured parameters directly, so their
    ``temperatures`` and ``weights`` are ``None``.
    ``abundance_values`` stores an immutable numerical baseline (linear number
    abundances relative to hydrogen), separate from fitted multipliers. Custom
    compositions remain usable after XSPEC's current table changes.
    """

    name: str
    model_type: str
    parameters: object
    abundance_table: object
    xset: object
    temperatures: np.ndarray | None = None
    weights: np.ndarray | None = None
    abundance_values: object = None

    def __post_init__(self):
        object.__setattr__(
            self, "parameters", MappingProxyType(dict(self.parameters))
        )
        object.__setattr__(
            self, "xset", MappingProxyType(dict(self.xset or {}))
        )
        if self.abundance_values is not None:
            object.__setattr__(self, "abundance_values",
                               MappingProxyType(_abundance_values(self.abundance_values)))

        if (self.temperatures is None) != (self.weights is None):
            raise ValueError("captured temperatures and weights must both be set or both be None")
        if self.temperatures is not None:
            temperatures = np.array(self.temperatures, dtype=float, copy=True)
            weights = np.array(self.weights, dtype=float, copy=True)

            if temperatures.ndim != 1 or weights.ndim != 1:
                raise ValueError("captured temperatures and weights must be 1D")
            if temperatures.size == 0 or temperatures.size != weights.size:
                raise ValueError(
                    "captured temperatures and weights must be nonempty "
                    "and equal-length"
                )
            if not np.isfinite(temperatures).all() or np.any(temperatures <= 0):
                raise ValueError("captured temperatures must be finite and positive")
            if not np.isfinite(weights).all():
                raise ValueError("captured weights must be finite")

            temperatures.setflags(write=False)
            weights.setflags(write=False)
            object.__setattr__(self, "temperatures", temperatures)
            object.__setattr__(self, "weights", weights)


def _get_source(model, dataset_id, session):
    if model is not None:
        if dataset_id is not None or session is not None:
            raise ValueError("supply a model or a dataset/session, not both")
        return model

    if session is None:
        from sherpa.astro import ui as session

    if dataset_id is None:
        return session.get_source()
    return session.get_source(dataset_id)


def _find_components(model):
    found = []
    visited = set()

    def visit(part):
        if id(part) in visited:
            return
        visited.add(id(part))

        model_type = type(part).__name__.lower()
        if model_type in _SUPPORTED_TYPES:
            found.append((part, model_type))
            return

        for child in getattr(part, "parts", ()):
            visit(child)

    visit(model)

    if not found:
        raise ValueError(
            "source contains no supported XSPEC CIE, NEI, or PShock "
            "components; Sedov models are not supported"
        )

    return found


def _single_temperature(model_type, parameters):
    parameter_name = _SINGLE_TEMPERATURE_PARAMETERS.get(model_type)
    if parameter_name is None:
        return None

    values = {
        str(name).lower(): value
        for name, value in parameters.items()
    }
    value = values.get(parameter_name.lower())
    if value is None:
        return None
    temperature = float(value)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError(
            f"{parameter_name} must be finite and positive for a "
            f"single-temperature CIE component ({model_type})"
        )
    return np.array([temperature]), np.array([1.0])


def _parameter(parameters, name):
    return next(
        (value for key, value in parameters.items()
         if str(key).lower() == name.lower()),
        None,
    )


def _validate_nei_parameters(model_type, parameters, component_name):
    temperature = _parameter(parameters, "kT")
    if temperature is None or temperature <= 0:
        raise ValueError(f"{component_name} ({model_type}) requires positive kT")
    if model_type in _PSHOCK_TYPES:
        lower = _parameter(parameters, "Tau_l")
        upper = _parameter(parameters, "Tau_u")
        if lower is None or upper is None or lower < 0 or upper <= lower:
            raise ValueError(
                f"{component_name} ({model_type}) requires 0 <= Tau_l < Tau_u"
            )
        return
    tau = _parameter(parameters, "Tau")
    if tau is None or tau < 0:
        raise ValueError(f"{component_name} ({model_type}) requires nonnegative Tau")
    if _NEI_INITIAL_POPULATIONS[model_type] == "initial_temperature":
        initial = _parameter(parameters, "kT_init")
        if initial is None or initial <= 0:
            raise ValueError(
                f"{component_name} ({model_type}) requires positive kT_init"
            )


def _make_edges(edges):
    if edges is None:
        edges = np.geomspace(0.01, 100.0, 1001)

    edges = np.asarray(edges, dtype=float)
    if (
        edges.ndim != 1
        or edges.size < 2
        or not np.isfinite(edges).all()
        or np.any(edges <= 0)
        or np.any(np.diff(edges) <= 0)
    ):
        raise ValueError(
            "edges must be finite, positive, strictly increasing energy "
            "boundaries in keV"
        )
    return edges


def _capture_cie_evaluation(component, model_type, parameter_vector, lo, hi,
                            bridge, library_path):
    """Evaluate one trial vector and capture its fresh native DEM metadata."""
    parameter_vector = np.asarray(parameter_vector, dtype=float)
    if (parameter_vector.ndim != 1
            or len(parameter_vector) != len(component.pars)
            or not np.isfinite(parameter_vector).all()):
        raise ValueError(f"invalid trial parameters in {component.name}")
    model_calc = getattr(component, "calc", None)
    if model_calc is None:
        raise TypeError(
            f"{component.name} does not expose Sherpa's XSPEC calc method"
        )
    parameters = {par.name: float(value)
                  for par, value in zip(component.pars, parameter_vector)}
    flux = None

    def evaluate():
        nonlocal flux
        component.cache_clear()
        flux = np.asarray(model_calc(parameter_vector, lo, hi), dtype=float)
        if flux.shape != lo.shape or not np.isfinite(flux).all():
            raise ValueError(f"native evaluation returned invalid flux for {component.name}")
        return flux

    with _CAPTURE_LOCK:
        try:
            temperatures, weights = bridge.capture(library_path, evaluate)
        except RuntimeError as exc:
            if "did not publish fresh temperature weights" not in str(exc):
                raise
            fallback = _single_temperature(model_type, parameters)
            if fallback is None:
                raise RuntimeError(
                    f"{component.name} ({model_type}) did not publish "
                    "temperature weights and has no unambiguous "
                    "single-temperature fallback"
                ) from exc
            temperatures, weights = fallback
    if flux is None:
        raise RuntimeError(f"native evaluation was not performed for {component.name}")
    return flux, temperatures, weights


def capture_plasma_model(
    model=None, *, dataset_id=None, session=None, edges=None
):
    """Capture each supported XSPEC CIE, NEI, or PShock component separately.

    Components are evaluated directly with their current Sherpa parameter
    values. The callback invokes the component's Sherpa ``calc`` method, and
    the bridge clears XSPEC's DEM metadata before evaluation.

    Parameters
    ----------
    model : Sherpa model, optional
        Source expression to inspect.
    dataset_id, session : optional
        Used to obtain a source from a Sherpa session when ``model`` is omitted.
    edges : array-like, optional
        Strictly increasing energy-bin boundaries in keV. Defaults to a
        logarithmic 0.01--100 keV grid.

    Returns
    -------
    tuple[CapturedPlasmaComponent, ...]
        One detached capture per unique supported component, in expression
        order. Expression-level scale factors are not applied.

    Notes
    -----
    CIE arrays are raw XSPEC metadata except for the documented single-
    temperature fallback. NEI/PShock components capture their model
    parameters without evaluating the native model. Neither path captures
    an absolute line flux.
    """
    source = _get_source(model, dataset_id, session)
    components = _find_components(source)

    try:
        from sherpa.astro import xspec
    except ImportError as exc:
        raise ImportError(
            "capture requires a Sherpa/XSPEC environment"
        ) from exc

    cie_components = any(model_type in _CIE_TYPES for _, model_type in components)
    if cie_components:
        edges = _make_edges(edges)
        lo, hi = edges[:-1], edges[1:]
        try:
            from sherpa.astro.xspec import _xspec
            from . import _xspec_bridge
        except ImportError as exc:
            raise ImportError(
                "build the optional XSPEC bridge for CIE component capture"
            ) from exc
        library_path = _xspec.__file__
    else:
        lo = hi = library_path = None

    captured = []

    # This serializes this module's own capture calls. Avoid concurrent XSPEC
    # model evaluations elsewhere while reading/restoring FunctionUtility state.
    with _CAPTURE_LOCK:
        abundance_table = xspec.get_xsabund()
        abundance_values = _xspec_abundance_values()
        xset = xspec.get_xsxset()

        for component, model_type in components:
            component_name = component.name
            parameters = {
                par.name: float(par.val)
                for par in component.pars
            }
            parameter_vector = np.asarray(
                list(parameters.values()), dtype=float
            )

            if not np.isfinite(parameter_vector).all():
                raise ValueError(f"nonfinite parameter in {component_name}")

            if model_type in _NEI_TYPES or model_type in _PSHOCK_TYPES:
                _validate_nei_parameters(model_type, parameters, component_name)
                captured.append(
                    CapturedPlasmaComponent(
                        name=component_name,
                        model_type=model_type,
                        parameters=parameters,
                        abundance_table=abundance_table,
                        xset=xset,
                        abundance_values=abundance_values,
                    )
                )
                continue

            _, temperatures, weights = _capture_cie_evaluation(
                component, model_type, parameter_vector, lo, hi,
                _xspec_bridge, library_path,
            )

            captured.append(
                CapturedPlasmaComponent(
                    name=component_name,
                    model_type=model_type,
                    parameters=parameters,
                    abundance_table=abundance_table,
                    xset=xset,
                    temperatures=temperatures,
                    weights=weights,
                    abundance_values=abundance_values,
                )
            )

    return tuple(captured)

