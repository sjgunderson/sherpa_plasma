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

The [database validation example](../examples/validate_atomdb.py) compares
local emissivity results, records input hashes, and fails if required data
checks are skipped.
