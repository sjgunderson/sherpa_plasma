# Local plasma database access

`AtomDB` reads an existing APED/AtomDB installation through PyAtomDB. It
supports atomic line searches, transition and level details, CIE and NEI/shock
line emissivities, and ratios. `read_plasma_model` reads existing built-in
APEC components in Sherpa. This package provides no custom plasma fitting
models or spectrum-generation kernels.

## Installation and files

Inside your science environment:

```bash
python -m pip install -e '.[plasma]'
export ATOMDB=/absolute/path/to/atomdb
```

The optional extra installs PyAtomDB and Astropy. Sherpa is needed only for
reading a loaded Sherpa model or using the fitting/plotting adapters.
No initialization or database downloads run automatically.

Atomic searches require a local `filemap` and its LA/LV atomic files.
Emissivity queries also require the CIE `apec_line.fits` and
`apec_coco.fits` tables. Versioned filenames can be supplied explicitly:

```python
from sherpa_plasma import AtomDB

db = AtomDB('/path/to/APED',
            linefile='apec_v3.0.9_line.fits',
            cocofile='apec_v3.0.9_coco.fits')
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
directly.

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

## Direct NEI and plane-parallel shock queries

The explicit-parameter APIs use PyAtomDB's `NEISession` and `PShockSession`.
Ionization age (`tau`, `tau_lower`, `tau_upper`) is `n_e * t` in cm^-3 s.
Temperatures are in keV unless `teunit='K'` is specified. `init_pop` may be
`'ionizing'` (initially neutral), `'recombining'` (initially fully ionized),
or a scalar initial equilibrium temperature in the selected temperature unit.
The NEI line-emissivity API broadcasts temperature and ionization-age arrays;
the PShock API integrates over its lower and upper age limits.

```python
with AtomDB('/path/to/APED') as db:
    nei = db.find_nei_lines(
        21.5, 22.2, element='O', ion=7, temperature=1.0,
        ionization_age=1.e11, init_pop='ionizing', brightest=5,
    )
    line = LineID.from_row(nei[0])
    epsilon = db.nei_line_emissivity(
        line, temperature=1.0, ionization_age=1.e11,
    )

    shock = db.find_pshock_lines(
        21.5, 22.2, element='O', ion=7, temperature=1.0,
        tau_lower=1.e10, tau_upper=1.e11, brightest=5,
    )
    shock_epsilon = db.pshock_line_emissivity(
        line, temperature=1.0, tau_lower=1.e10, tau_upper=1.e11,
    )
```

Returned `Epsilon` values are photons cm^3/s and include the selected
abundance baseline and multipliers. Missing local NEI tables, unsupported
initial populations, and invalid age bounds raise errors; no downloads or
unmatched-model approximations are performed. Sherpa's bundled eigenvector
table is preferred; if it is unavailable, the local ion-balance data is
preloaded through the configured filemap in offline mode. Missing entries are
reported instead of falling back to PyAtomDB preferences or download handling.

To rank AtomDB lines using the total currently active supported Sherpa/XSPEC
CIE, NEI, or PShock model, omit `kT` and pass `brightest`:

```python
with AtomDB('/path/to/APED') as db:
    rows = db.find_lines(6.0, 25.0, unit='Angstrom', brightest=20)
    for row in rows:
        print(row['Lambda'], row['Element'], row['Ion'],
              row['ModelContribution'])
```

By default, the source is read from the Sherpa UI session. Use
`model=source_expression` or `dataset_id=...` and optionally `session=...` to
choose another source. `element`, `ion`, and `min_emissivity` can filter the
ranked results; `min_emissivity` applies to `ModelContribution`. Abundances
and abundance tables are taken from the captured components. The established
`Epsilon` field remains NaN in this mode; the summed ranking is exposed in
`ModelContribution`. Supplying `kT` retains the existing single-temperature
search and ranks by `Epsilon` when `brightest` is used. Without either `kT` or
`brightest`, `find_lines` remains an atomic-only LA search.

## Read an existing Sherpa plasma model

```python
from sherpa.astro import ui
from sherpa_plasma import read_plasma_model

# Read your current source, including a shared-grid source wrapper.
states = read_plasma_model(dataset_id=1)
for state in states:
    print(state.name, state.kT, state.redshift, state.norm)
    print(state.parameters, state.abundance_table, state.xset)

# Or supply a component/expression or a separate Session explicitly.
states = read_plasma_model(ui.get_source(1))
# states = read_plasma_model(dataset_id=1, session=my_session)

with AtomDB('/path/to/APED') as db:
    state = db.read_model(dataset_id=1)[0]
    rows = db.find_lines(21.5, 22.2, element='O', kT=state.kT)
```

Supported components are `xsapec`, `xsbapec`, `xsvapec`, `xsbvapec`,
`xsvvapec`, and `xsbvvapec`. Compound expressions and shared-grid wrappers
are traversed in expression order; each unique component is returned once.
Other components, such as absorption and continua, are ignored. An expression
without a supported APEC component raises an error.

The returned `PlasmaState` snapshots preserve all component parameters,
resolve current linked values, and record XSPEC's abundance-table name and
XSET settings. Reading does not evaluate a model, alter parameters, change
session sources, or open database tables. Reread after fitting or changing
parameters/settings. `state.velocity` is Gaussian sigma in km/s;
`state.metal_abundance` exposes scalar `Abundanc`, while `state.abundances`
contains only explicitly parameterized elemental abundances.

Database queries use rest-frame wavelengths/energies. The example above uses
only the model's temperature: it does not reproduce its abundances, redshift,
normalization, absorption, broadening, or complete spectrum. XSPEC and the
local database may use different versions and abundance tables. Scalar factors
outside a component do not modify its recorded normalization.

## Supported CIE model capture and weighted contributions

Capture recognizes Sherpa/XSPEC CIE line-emission families: APEC variants
(including velocity-broadened, variable-abundance, two-temperature,
resonant-scattering, and supernova variants), `cie`/`equil`, `mekal`,
`raymond`, differential-emission-measure families (`cemekl`, `cevmkl`,
`cempow`, `cheb6`, `expcheb6`, `gadem`, and `wdem`), and cooling-flow/CPh
families. For models that publish temperature/weight arrays, the XSPEC bridge
uses those arrays. For supported single-temperature models that publish no
arrays, capture uses their line temperature with unit weight; for `tapec`
that is `kTi`. A multi-temperature model with no published arrays fails
explicitly rather than being approximated by a single temperature.

The line search uses AtomDB's CIE emissivities at the captured temperature
grid. This is a weighted AtomDB line contribution, not a reconstruction of
the native XSPEC spectrum: models using MEKAL, Raymond-Smith, SPEX, different
atomic-data versions, or additional model settings can have different line
emissivities. The `nlapec` XSPEC model is continuum-only and is not included.

## Supported XSPEC NEI and shock captures

`capture_plasma_model` and `query_plasma_lines` also support model families
that map directly to PyAtomDB's NEI or PShock interfaces:

- `nei`, `vnei`, and `vvnei` use their captured electron temperature and
  ionization age with an initially ionizing population.
- `gnei`, `vgnei`, `vvgnei`, `rnei`, `vrnei`, and `vvrnei` use the captured
  initial temperature as PyAtomDB's initial equilibrium population, followed
  by evolution at the captured electron temperature and ionization age.
- `pshock`, `vpshock`, and `vvpshock` use the captured electron temperature
  and lower/upper ionization-age bounds with PyAtomDB's plane-parallel shock
  integration.

Their abundance parameters and XSPEC abundance table are applied in the same
way as for captured CIE components. Capture snapshots their parameters without
evaluating the NEI/shock model or altering the Sherpa source. Results are
AtomDB weighted emissivity contributions, not XSPEC fluxes or absolute line
fluxes; model normalization, redshift, absorption, and broadening are not
applied. `Sedov` and other XSPEC models without a direct PyAtomDB mapping are
not supported and are not approximated.

The same `brightest=` model-ranking option on `find_lines` and the component
breakdown from `query_plasma_lines` work for these supported captured models.

`query_plasma_lines` remains available when the per-component breakdown is
needed. It reads and captures the current source from the default Sherpa UI
session internally, searches the requested interval at each captured
temperature or NEI/shock state, takes the union of transitions found, and
returns each component's emissivity contribution and the sum across
components:

```python
from sherpa_plasma import AtomDB, LineID

with AtomDB('/path/to/APED') as db:
    result = db.query_plasma_lines(
        6.0, 25.0, unit='Angstrom', brightest=20
    )

    for index, row in enumerate(result.lines):
        line = LineID.from_row(row)
        per_component = [
            (component.name, component.values[index])
            for component in result.components
        ]
        print(line, row['Lambda'], per_component, result.total[index])
```

The same source-selection options described for `find_lines` apply. With no
source selector, the default Sherpa UI session's current source is used.

For reusable snapshots (for example, to inspect or compare captures after
further fitting changes), call `capture_plasma_model` explicitly and pass its
result with `components=`:

```python
from sherpa_plasma import capture_plasma_model

captured = capture_plasma_model(dataset_id=1)
with AtomDB('/path/to/APED') as db:
    reference = db.query_plasma_lines(
        6.0, 25.0, unit='Angstrom', components=captured
    )
```

This explicit path uses the detached temperature, weight, parameter, and
abundance snapshots as captured; it does not reevaluate the Sherpa model.
Multi-model comparison/reporting is not yet provided by the query result.

The optional `brightest=N` keeps the N lines ranked by the total contribution
summed across all captured components. If omitted, all discovered lines are
returned. Components retain capture order; each component's `values` and
`result.total` align with `result.lines`. `result.lines['ModelContribution']`
matches `result.total`, while `Epsilon` remains NaN.

For a CIE component *c* and line *l*, the reported quantity is
`sum_i(weight[c, i] * AtomDB_emissivity(l, temperature[c, i]))`. NEI
components use their captured temperature, ionization age, and initial
population; PShock components use their captured temperature and age bounds.
Each component's abundance settings are applied. CIE weights are raw XSPEC
metadata; all results are emissivity contributions, **not absolute line
fluxes**. No model normalization, expression scale factor, redshift,
absorption, or broadening is applied.

Abundance handling follows the ISIS/APED separation between a baseline
abundance table and per-element multipliers. The captured XSPEC table is mapped
to the corresponding local AtomDB table (`angr`→`AG89`, `aspl`→`Asplund09`,
`feld`→`Feldman`, `aneb`→`AE82`, `grsa`→`GS98`, `wilm`→`Wilm00`,
`lodd`→`Lodd03`). A scalar `Abundanc` parameter scales metals; explicit
elemental parameters override their corresponding factors. A zero abundance
excludes that element's lines from the discovered set (while another
component with nonzero abundance can still contribute those same lines).
Unsupported XSPEC abundance tables and mapped tables missing from the local
ABUND file raise an error instead of silently using a different baseline.
This table matching does not establish absolute-flux equivalence between
XSPEC and AtomDB; check the normalization and other model conventions before
making flux claims.
The table mapping follows the [PyAtomDB XSPEC wrapper](https://github.com/AtomDB/pyatomdb/blob/master/pyatomdb/wrappers/kappa_xspec.py);
the baseline-plus-element-multiplier design mirrors ISIS's
[`set_abund`](https://github.com/houckj/isis/blob/master/share/emis-cmds.sl)
and [`aped_set_abund`](https://github.com/houckj/isis/blob/master/contrib/aped_fit_models.sl)
APED workflow. Additional XSPEC XSET abundance overrides are recorded in the
capture but are not interpreted by this query.

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
