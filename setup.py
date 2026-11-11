"""Optional build hook for the XSPEC state-capture and line-profile bridges."""
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
    library_candidates = [include.parent / "lib"]
    if os.environ.get("CONDA_PREFIX"):
        library_candidates.append(Path(os.environ["CONDA_PREFIX"]) / "lib")
    libdir = next((path for path in library_candidates if
                   any(path.glob("libXSFunctions.*"))), None)
    if libdir is None:
        raise RuntimeError("Cannot locate libXSFunctions for the XSPEC profile bridge")
    extensions.append(Extension(
        "sherpa_plasma._xspec_profiles",
        ["src/sherpa_plasma/_xspec_profiles.cpp"],
        include_dirs=[str(include)],
        library_dirs=[str(libdir)],
        libraries=["XSFunctions", "XSUtil"] + (["dl"] if sys.platform == "linux" else []),
        extra_link_args=[f"-Wl,-rpath,{libdir}"],
        extra_compile_args=["-std=c++11"],
        language="c++",
    ))

setup(ext_modules=extensions)
