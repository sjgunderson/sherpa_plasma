# Running the tests

Install the package's test dependencies and run the test suite from this
project directory:

```bash
python -m pip install -e '.[test]'
python -m pytest
```

Core tests do not need a database or Sherpa. Database integration tests use
the local AtomDB directory named by `SHERPA_PLASMA_ATOMDB`:

```bash
export SHERPA_PLASMA_ATOMDB=/absolute/path/to/atomdb
# Optional versioned filenames:
# export SHERPA_PLASMA_ATOMDB_LINEFILE=apec_v3.0.9_line.fits
# export SHERPA_PLASMA_ATOMDB_COCOFILE=apec_v3.0.9_coco.fits
python -m pytest
```

Tests never download AtomDB. XSPEC capture tests require the optional compiled
bridge, built in a Sherpa/XSPEC environment:

```bash
SHERPA_PLASMA_BUILD_XSPEC_BRIDGE=1 python -m pip install -e .
```

The build also creates the native profile bridge and links it against
`libXSFunctions` from the same XSPEC installation. If the headers are not
under `HEADAS/include` or `CONDA_PREFIX/include`, set
`SHERPA_PLASMA_XSPEC_INCLUDE` to their include directory; the matching
library must be in its sibling `lib` directory.

The XSPEC capture tests include a narrow reconstruction check for default
abundances in `wdem` and `mkcflow`: captured temperature weights reproduce
the corresponding weighted single-temperature APEC spectrum for the tested
parameters. This does not establish their interpretation for other abundance
settings or model families. Separate isolated-line tests validate the
experimental `XSwdem` corrections for each calibrated ion and selected
`XSvwdem`/`XSvvwdem` abundance cases under the restrictions described in
[UV_PUMPING.md](UV_PUMPING.md). The fitted dimensionless `p` controls the
temperature-integrated ratio; it does not represent a fixed physical rate.
The reference tests can use either bundled 3.0.9 or 3.1.3 calibration.
`test_triplet_installed.py` also checks an unbundled synthetic installation,
including file matching, derived cascade branching, and DEM integration.

All tests, including the uploaded UV regressions, live under `tests/`.
`tests/test_abundance_compositions.py` checks general XSPEC table names,
custom mappings/vectors/files, separate fitted multipliers, and isolation
across CIE, NEI, shock, ratios, model ranking, and cached DEM spectra using
controlled data. It also checks immutable numerical composition snapshots and
zero-abundance discovery. `tests/test_atomdb_real.py` includes an optional
comparison of all built-in XSPEC tables and custom compositions against an
independent PyAtomDB session. This requires local AtomDB data and Sherpa/XSPEC:

```bash
SHERPA_PLASMA_ATOMDB=/absolute/path/to/APED \
python -m pytest tests/test_atomdb_real.py -q
```

`tests/test_uv_coverage.py` exercises independent sparse transitions across the
full DEM, interpolation across missing endpoints, endpoint clamping, absolute
weight preservation, and absent triplets without requiring Sherpa.
`tests/test_uv_model_coverage.py` checks correction and diagnostics over the
same full distribution, caching, and broadened redistribution into a missing
branch with controlled doubles. These are not native XSPEC validation.
`tests/test_uv_profiles_bridge.py` compiles the production C++ bridge against
a small controlled Aped interface to check sparse-record handling, both tails,
endpoint clamping, and donor profiles. This also does not validate XSPEC's
actual ABI or broadened line physics.
`tests/test_uv_abundances.py` checks named and custom table acceptance for all
supported wrapper families, zero abundances, numerical custom-vector changes,
and rebuilding with a cached child. Compiled bridge tests check that native
solar scaling is applied once, including the donor profile for an empty branch.
These use controlled interfaces and do not establish real XSPEC compatibility.
The compiled bridge tests also exercise `APECTHERMAL` overrides, native-temperature
thermal widths, serial/threaded agreement, and projection linearity with a
controlled Gaussian implementation and actual C++ worker threads. Wrapper tests
cover the two XSETs for all supported families, cache reuse, option-change
invalidation, and stale-bridge rebuild errors.
`tests/test_uv_xsets_real.py` adds independent isolated-line comparisons for
thermal APEC, broadened APEC, wDEM, and fixed-weight `mkcflow` references with
the new options. It also compares native serial and multithreaded spectra and
checks the resolved XSPEC controls and diagnostic restoration. These new
scientific-backend comparisons require Sherpa/XSPEC and both rebuilt bridges;
they were not run in the development workspace, which lacks those backends.

```bash
SHERPA_PLASMA_UV_XSPEC_ROOT=/absolute/path/to/apec_v3.0.9 \
python -m pytest tests/test_uv_xsets_real.py -q
```

The real abundance regressions cover the built-in tables and a custom file for
single-temperature APEC, isolated line profiles for `wdem`, `bvwdem`, and
`mkcflow` under `lodd` and a custom table, and custom-vector reload detection.
The real-model tests exercise `Tmax=1`, `beta=.1`, `p_Si=1` for all six wDEM
variants, formerly rejected sodium configurations, and broad native-grid line
projections. Rebuild both optional bridges before running in CIAO:

```bash
SHERPA_PLASMA_BUILD_XSPEC_BRIDGE=1 python -m pip install -e .
```

Run the full suite in the CIAO environment:

```bash
SHERPA_PLASMA_UV_XSPEC_ROOT=/opt/anaconda3/envs/ciao-4.18/spectral/modelData/apec_v3.0.9 \
HEADAS=/opt/anaconda3/envs/ciao-4.18/spectral PYTHONPATH=src \
/opt/anaconda3/envs/ciao-4.18/bin/python -m pytest tests/ -q
```

Use the `spectral` subdirectory for `HEADAS` so XSPEC can read its abundance
tables. Adapt these paths to the installed environment. A run without an APEC
root skips the real UV model tests; it is not XSPEC validation.

For an isolated-triplet investigation with the verified APEC files and the
XSPEC bridge, run:

```bash
python examples/characterize_uv_dem.py --root "$SHERPA_PLASMA_UV_XSPEC_ROOT" \
  --model wdem --parameters '{"Tmax":0.8,"beta":0.5,"Redshift":0.02,"abundanc":0.6}'
python examples/characterize_uv_dem.py --root "$SHERPA_PLASMA_UV_XSPEC_ROOT" \
  --model vwdem --parameters '{"Tmax":0.8,"beta":0.5,"Redshift":0.02,"Na":0.6}'
python examples/characterize_uv_dem.py --root "$SHERPA_PLASMA_UV_XSPEC_ROOT" \
  --model bwdem --parameters '{"Tmax":0.8,"beta":0.5,"Redshift":0.02,"Velocity":0}'
python examples/characterize_uv_dem.py --root "$SHERPA_PLASMA_UV_XSPEC_ROOT" \
  --model mkcflow --parameters '{"lowT":0.25,"highT":0.8,"Redshift":0.02,"Abundanc":0.6}'
python examples/characterize_uv_dem.py --root "$SHERPA_PLASMA_UV_XSPEC_ROOT" \
  --model mkcflow --parameters '{"lowT":0.25,"highT":0.8,"Redshift":0.02,"Abundanc":0.6}' \
  --fixed-cooling-weights
```

The script copies the line file to a temporary directory, removes one
transition at a time, evaluates each copy in a fresh process, and reports
captured weights, isolated fluxes, and a candidate weighted-line prediction.
Predictions integrate every positive-weight temperature and treat unlisted
transitions as zero at their native endpoints.
It is a characterization tool, not a pass/fail validation. In particular,
removing a line from a cooling-flow database can change the cooling function
and therefore the temperature weights and flux across many bins; the raw
spectrum difference is then not an isolated-line flux. The `weights_unchanged`
and `nonzero_bins` fields expose that complication. In the tested `wdem`
case, the scalar abundance also applies to sodium, unlike the default trace
abundance behavior of `XSapec`.
For broadened wDEM variants, the wrapper tests compare the native profile
bridge with independent isolated lines for all three model variants, zero
and nonzero velocities, truncated grids, and a grid containing tails but not
line centers. Tests also check every native temperature for several ions,
including the final grid endpoint, and confirm preservation of XSPEC
diagnostic values. Even at zero velocity, the tested broadened lines span
multiple bins. Use `--include-profile` to print the isolated per-bin differences.
The fixed-weight option avoids changing the cooling function: it first verifies
that a weighted APEC sum reproduces the original cooling-flow spectrum, then
isolates lines in separate variable-abundance APEC processes using the
original captured weights and the corresponding elemental abundances.

The [database validation example](../examples/validate_atomdb.py) compares
local emissivity results, records input hashes, and fails if required data
checks are skipped.

The experimental UV-pumping line adapter can be compared independently with
compiled XSPEC using copied APEC files in separate processes. This check
requires Sherpa/XSPEC, Astropy, a working `HEADAS`, and an APEC root containing
both `_line.fits` and `_coco.fits`:

```bash
export HEADAS=/opt/anaconda3/envs/ciao-4.18/spectral
export SHERPA_PLASMA_UV_XSPEC_ROOT=/absolute/path/to/apec_v3.0.9
python -c 'import astropy; import pytest; raise SystemExit(pytest.main(
    ["-q", "tests/test_uv_xspec_real.py", "tests/test_uv_model_real.py"]))'
```

The line tests isolate C V, N VI, O VII, Ne IX, Na X and Mg XI. The composite
tests additionally compare each candidate ion against independently isolated
XSPEC triplet lines at two temperatures, and include a synthetic fit,
abundance tables, redshift, a nonuniform grid, and the supported APEC
trace-abundance cases. For Na X and Mg XI, the lower and upper native-triplet
coverage endpoints are also compared with isolated XSPEC lines. These endpoint
checks passed for the reference 3.0.9 and 3.1.3 files. They do not validate all
global XSPEC settings or a particular dataset merely from its filename.

The tests also compare the selected forbidden and J=1 intercombination line
energies for C, N, O, Ne, Mg and Si to the rounded literature values in
[Table 1](https://arxiv.org/abs/astro-ph/0012218), using a 0.1% tolerance to
allow for differences between atomic datasets. This checks line identities;
it is not an independent validation of the pumping-rate calibration.
Separately, `test_uv_literature.py` compares the forbidden decay rate,
statistical effective branching, and their ratio for those six ions against
[Porquet and Dubau (2000), Table 2](https://arxiv.org/abs/astro-ph/0002319)
with a 25% tolerance. Their older atomic calculations give forbidden rates
that differ from the packaged AtomDB rates by up to about 18%, and effective
branching by up to about 17%; this is a coarse cross-check, not evidence of
precision accuracy or a complete comparison of pumping physics.
An independent steady-state four-level rate-network test also checks the
transfer and redistribution for zero through strong pumping, including a
substantial radiative cascade between triplet P levels.

The reference tool can also be run directly:

```bash
python examples/validate_uv_xspec.py --root "$SHERPA_PLASMA_UV_XSPEC_ROOT"
```

Use the root corresponding to the actual XSPEC data, not just a filename with
a matching version. During development, the test passed against the CIAO
bundled 3.0.9 data and independently downloaded 3.1.3 data, with `angr`
abundances and the default unbroadened settings. These tests are not evidence
that a separate local installation has identical contents.

### Checking your installation from a notebook

Run this cell from the same Python environment as Sherpa. Replace both paths
with the project directory and your own APEC root; the root omits the
`_line.fits` and `_coco.fits` suffixes. Set `HEADAS` before starting the
notebook, or replace its value in the cell with your CIAO spectral directory.
The tests run in a fresh Python process so that earlier XSPEC calls in the
notebook cannot leave cached APEC data in the test process.

```python
import os
from pathlib import Path
import subprocess
import sys

project = Path("/absolute/path/to/sherpa_plasma")
apec_root = "/absolute/path/to/apec_v3.1.3"
env = os.environ.copy()
env["HEADAS"] = "/opt/anaconda3/envs/ciao-4.18/spectral"
env["SHERPA_PLASMA_UV_XSPEC_ROOT"] = apec_root

subprocess.run(
    [
        sys.executable, "-m", "pytest", "-q",
        "tests/test_uv_model_real.py",
        "-k", "zero_pumping_preserves_child_and_parameter_identity or "
              "pumped_sodium_conserves_isolated_photons or "
              "temperature_coverage_endpoints",
    ],
    cwd=project, env=env, check=True,
)
```

Six tests should pass. Wrapper construction checks both file hashes against
the packaged registry; a hash mismatch means those files have not been
validated, even if they have the same version label. For the complete
independent comparison, remove the `-k` argument and its selection string
and add `tests/test_uv_xspec_real.py` to the command. These tests copy APEC
data into temporary directories and launch separate XSPEC processes; they
do not modify the installed data. A failure raises
`subprocess.CalledProcessError` and prints the pytest diagnostics.

An experimental atomic-data inventory can be generated without modifying or
downloading into an AtomDB installation:

```bash
python examples/inventory_uv_atomic.py \
    --filemap /path/to/atomdb/filemap_v3.0.9 \
    --atomic-root /path/to/atomdb \
    --line-file /path/to/atomdb/apec_v3.0.9_line.fits
```

The inventory derives radiative inputs from the version-matched filemap and
records hashes and native-grid coverage. `candidate_unvalidated` entries are
not declarations of supported ions. Missing transitions and significant
additional radiative channels are reported rather than silently assumed
away. Cascades among the triplet P levels are included explicitly. Other
channels are rejected if their rate exceeds 1e-12 of the modeled decay rate
of a level; the retained fractions are recorded in the calibration.
Save the inventory as JSON and generate a manifest using the matching
continuum and line files:

```bash
python examples/build_uv_registry.py \
    --inventory /path/to/inventory.json \
    --line-file /path/to/atomdb/apec_v3.0.9_line.fits \
    --continuum-file /path/to/atomdb/apec_v3.0.9_coco.fits \
    --version 3.0.9
```

Generating a manifest alone does not validate its candidates or establish
compatibility with the packaged registry.

The experimental warm-cache benchmark uses independent, identically
configured bare and wrapped child models. Its child-call count checks that
pumping-only trial vectors reuse the child spectrum:

```bash
python examples/benchmark_uv.py --root "$SHERPA_PLASMA_UV_XSPEC_ROOT" \
    --bins 10000 --iterations 21
python examples/benchmark_uv.py --root "$SHERPA_PLASMA_UV_XSPEC_ROOT" \
    --bins 100000 --iterations 21
```

One serial run with the CIAO 4.18 bundled 3.0.9 data measured mean bare and
wrapped changing-temperature calls of 25.37 and 23.06 ms at 10,000 bins, and
216.29 and 214.79 ms at 100,000 bins. Pumping-only sequences made one child
`calc` call and no native-line reads over 21 trials. Python-traced peak
allocations for those sequences were about 0.68 MB and 6.44 MB respectively;
this does not measure all native XSPEC memory. Wrapper initialization took
about 0.85 and 0.83 seconds in the separate processes. These are measurements
for one setup, not a performance guarantee.

An exploratory Na X `XSbvvapec` test with velocity sigma 1000 km/s found a
maximum per-bin discrepancy of about `9.77e-6` of the total isolated-line
flux when compared with a velocity-only analytic Gaussian. A zero-velocity
`XSbvvapec` reference was also not deposited in a single bin. Neither
observation establishes an exact broadening profile. Thermal and broadened
APEC/wDEM corrections now use native XSPEC projection. The newly supported
APEC variants and XSET combinations require the native regressions above.

To measure the zero-velocity `XSbvvapec` lines without assuming a profile:

```bash
python examples/validate_uv_xspec.py --root "$SHERPA_PLASMA_UV_XSPEC_ROOT" \
    --model bvvapec --velocity 0 --profile measure
```

In the tested CIAO 4.18 setup, each Na X triplet line spans 16 bins on the
default energy grid even at zero velocity. This is a measurement, not an
analytic prescription for the profile.

The public HEASoft 6.35 source explains why a velocity-only Gaussian is not
an adequate reference: [`bvvapec.cxx`](https://heasarc.gsfc.nasa.gov/FTP/software/package_managers/source/heasoft-6.35/Xspec/src/XSFunctions/bvvapec.cxx)
enables thermal broadening as well, and
[`Aped.cxx`](https://heasarc.gsfc.nasa.gov/FTP/software/package_managers/source/heasoft-6.35/Xspec/src/XSFunctions/Aped.cxx)
computes thermal widths at the native temperature records. Its
[`calcLines.cxx`](https://heasarc.gsfc.nasa.gov/FTP/software/package_managers/source/heasoft-6.35/Xspec/src/XSFunctions/calcLines.cxx)
uses a tabulated fast profile and a configurable tail cutoff. The wDEM bridge
uses the native APEC line projection, and independent isolated-line comparisons
exercise its profiles and truncated energy grids. `APECMINFLUX` and
`LINECRITLEVEL` settings are rejected; source inspection alone is not validation.

