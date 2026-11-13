"""Local plasma-database access and Sherpa plasma-state inspection."""

import importlib

from .plasma import (AtomDB, LineID, ComponentLineContributions,
                     PlasmaLineContributions)
from .plasma_state import PlasmaState, read_plasma_model
from .xspec_capture import CapturedPlasmaComponent, capture_plasma_model

__version__ = "0.1.0"
__all__ = [
    "AtomDB", "LineID", "ComponentLineContributions",
    "PlasmaLineContributions", "PlasmaState", "read_plasma_model",
    "CapturedPlasmaComponent", "capture_plasma_model",
    "CompiledCIEModel", "compile_cie_model",
    "AtomDBDEMModel", "compile_dem_function",
    "wrap_triplet_pumping", "AtomDBTripletProvider",
]


def __getattr__(name):
    if name in {"CompiledCIEModel", "compile_cie_model"}:
        module = importlib.import_module(".cie_model", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in {"AtomDBDEMModel", "compile_dem_function"}:
        module = importlib.import_module(".dem_model", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name == "wrap_triplet_pumping":
        module = importlib.import_module("._uv_model", __name__)
        value = module.wrap_triplet_pumping
        globals()[name] = value
        return value
    if name == "AtomDBTripletProvider":
        module = importlib.import_module("._atomdb_pumping", __name__)
        value = module.AtomDBTripletProvider
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
