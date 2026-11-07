# sherpa_plasma

`sherpa_plasma` is a user contributed package for advanced plasma
modeling of X-ray data for use in [Sherpa](https://github.com/sherpa/sherpa),
the recommended X-ray modeling package for the Chandra X-ray Observatory.
This package provides access for browsing a local installation of APED/AtomDB to
query line emissivities, capturing plasma models (e.g., `apec` and `nei`
varients) details, and identifying lines present in a modeled X-ray spectrum.

The methods used in this package are based on the (open source)
Interactive Spectral Interpretation Software ([ISIS](https://github.com/houckj/isis);
[Houck & Denicola 2000](https://scixplorer.org/abs/2000ASPC..216..591H/abstract)).

## Install

```bash
python -m pip install '.[plasma]'
export ATOMDB=/absolute/path/to/atomdb
```

PyAtomDB and Astropy are optional extras and are needed for database access.
Sherpa/CIAO is needed only for reading a loaded Sherpa model or capturing
XSPEC plasma components. The optional compiled XSPEC bridge is built only
when `SHERPA_PLASMA_BUILD_XSPEC_BRIDGE=1` is set and XSPEC headers are
available.

```bash
export SHERPA_PLASMA_BUILD_XSPEC_BRIDGE=1
python -m pip install '.[plasma]'
export ATOMDB=/absolute/path/to/atomdb
```

## Quick start

```python
from sherpa_plasma import AtomDB, LineID

with AtomDB() as db:
    rows = db.find_lines(21.5, 22.2, element="O", ion=7)
    line = LineID.from_row(rows[0])
    print(db.line_info(line))
```

See `docs/PLASMA.md` for database setup, API details, and examples.

## Tests

```bash
python -m pip install '.[test]'
python -m pytest
```

Database integration tests require a local AtomDB installation; XSPEC capture
tests also require Sherpa/XSPEC and the optional bridge build.
