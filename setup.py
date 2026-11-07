"""Optional build hook for the XSPEC state-capture bridge."""
import os
from pathlib import Path
import sys

from setuptools import Extension, setup

extensions = []
if os.environ.get("SHERPA_PLASMA_BUILD_XSPEC_BRIDGE") == "1":
    if sys.platform not in ("linux", "darwin"):
        raise RuntimeError("The XSPEC state bridge currently supports Linux and macOS")
    candidates = [os.environ.get("SHERPA_PLASMA_XSPEC_INCLUDE")]
    for name in ("HEADAS", "CONDA_PREFIX"):
        if os.environ.get(name):
            prefix = Path(os.environ[name])
            candidates.extend([prefix / "include", prefix / "heasoft" / "include"])
    include = next((Path(p) for p in candidates if p and
                    (Path(p) / "XSFunctions/Utilities/FunctionUtility.h").is_file()), None)
    if include is None:
        raise RuntimeError("Set SHERPA_PLASMA_XSPEC_INCLUDE to the include directory of "
                           "the XSPEC installation used by Sherpa")
    extensions.append(Extension(
        "sherpa_plasma._xspec_bridge",
        ["src/sherpa_plasma/_xspec_bridge.cpp"],
        include_dirs=[str(include)],
        libraries=["dl"] if sys.platform == "linux" else [],
        extra_compile_args=["-std=c++11"],
        language="c++",
    ))

setup(ext_modules=extensions)
