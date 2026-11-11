# Local plasma database access

`AtomDB` reads an existing APED/AtomDB installation through PyAtomDB. It
supports atomic line searches, transition and level details, CIE and NEI/shock
line emissivities, and ratios. `read_plasma_model` reads existing built-in
APEC components in Sherpa. The database API does not itself provide an
AtomDB spectrum-generation kernel. For a custom AtomDB CIE spectrum integrated
over a user-defined differential emission measure, see
[CIE_MODEL.md](CIE_MODEL.md).

## Installation and files

Inside your science environment:

```bash
python -m pip install -e '.[plasma]'
export ATOMDB=/absolute/path/to/atomdb
```

The optional extra installs PyAtomDB and Astropy. Sherpa is needed for
`AtomDBDEMModel` and `CompiledCIEModel`, reading a loaded Sherpa model, or
using the fitting/plotting adapters.
No initialization or database downloads run automatically.

Atomic searches require a local `filemap` and its LA/LV atomic files.
Emissivity queries also require the CIE `apec_line.fits` and
`apec_coco.fits` tables. Versioned filenames can be supplied explicitly:

```python
from sherpa_plasma import AtomDB

db = AtomDB('/path/to/APED',
            linefile='apec_v3.1.3_line.fits',
            cocofile='apec_v3.1.3_coco.fits')
print(db.info)
```

Paths may be absolute or relative to the database root. The connection
loads tables lazily and reuses the open PyAtomDB session for subsequent
queries. Reuse a connection and call `db.close()` when finished, or use a
context manager. Temporary PyAtomDB caches are kept outside the database;
the user's database files and environment variables are preserved.

NEI and plane-parallel shock queries require the local `apec_nei_line.fits`
and `apec_nei_comp.fits` tables. Sherpa's bundled `eigen_v*.fits` table is
used for ion-balance eigenvectors when available; otherwise, the local
eigenvector data referenced by the AtomDB `filemap` is used. Override the NEI
table names with `nei_linefile=` and `nei_cocofile=` when constructing
`AtomDB`; these files and the required eigenvector data are read offline only
when an NEI or shock query is requested. PyAtomDB's optional `userdata`
preferences file is not needed for line calculations and is neither created
nor modified. Missing required local line or ion-balance data raises an error
directly. If a local set of eigenvector files can not be located, PyAtomDB will
request to download them into the (user supplied) subdirectory of their APED installation.


## Search lines and inspect levels

```python
from sherpa_plasma import AtomDB, LineID

with AtomDB('/path/to/APED') as db:
    rows = db.find_lines(21.5, 22.2, element='O', ion=7)
    line = LineID.from_row(rows[0])
    details = db.line_info(line)
    levels = db.levels('O', 7)
    transitions = db.transitions('O', 7)
```

Ion numbers are spectroscopic stages: O VII is element 8, ion 7. Level
identities are one-based; the first LV table row is level 1. Transition
identities are specific to the database version.

`find_lines` returns a detached NumPy structured array with `Element`,
`Ion`, `UpperLev`, `LowerLev`, `Lambda`, `Energy`, `Epsilon`,
`ModelContribution`, and `Einstein_A`. Wavelength is Angstrom, energy is
keV, `Epsilon` is a single-temperature emissivity in photons cm³/s,
`ModelContribution` is the weighted total for a captured model, and Einstein
A is s⁻¹. Unavailable fields are NaN.
Atomic searches prefer positive observed wavelengths when available.
Bounds are inclusive; `unit='keV'`, `'eV'`, or `'nm'` are also supported.

## Emissivity queries

```python
with AtomDB('/path/to/APED') as db:
    rows = db.find_lines(21.5, 22.2, element='O', ion=7,
                         kT=.2, brightest=3, min_emissivity=1e-20)
    lines = [LineID.from_row(row) for row in rows]
    temperatures = [.15, .2, .25]
    emissivity = db.line_emissivity(lines[0], temperatures)
    blend = db.line_emissivity(lines, temperatures)
    ratio = db.ratio_emissivity(lines[0], lines[1], temperatures)
```

`kT` selects the CIE emissivity tables rather than the atomic LA tables.
Use `teunit='K'` for temperatures in Kelvin. `db.temperature_range` and
`db.temperatures` expose the available range and native knots in keV.
Out-of-range temperatures raise an error. Queries at native knots use the
selected table directly; intermediate temperatures use PyAtomDB line-list
interpolation. Duplicate line identities are rejected. Missing transitions
have zero emissivity; a zero ratio denominator returns NaN.

`abundances={'O': .5, 'Fe': 1.2}` supplies per-element multipliers relative
to the active abundance table. `abundset` on an individual emissivity query
can select a different local table; otherwise the connection's configured
table is used (`abundset=None` retains the emissivity tables' native set).
Alternate sets need the local ABUND file referenced by `filemap`. Each query
resets element multipliers and the abundance table before applying its own.

## Validation

Controlled-backend tests exercise queries, units, isolation, and cleanup.
Optional local-data tests compare metadata, line emissivities, blends, and
ratios with an independent PyAtomDB session. The existing-model reader has
separate tests for compound sources, linked values, settings snapshots,
and avoiding model evaluation. Weighted captured-component queries also have
controlled tests for direct model selection, reusable snapshots, model
abundances, temperature grids, and top-N ranking. With Sherpa/XSPEC installed,
a parameterized capture test exercises each supported class present in that
build. The full Sherpa/XSPEC family matrix could not be run in environments
without Sherpa/XSPEC. See [TESTING.md](TESTING.md) for commands and
prerequisites.
