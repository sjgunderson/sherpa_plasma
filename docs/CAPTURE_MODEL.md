
# Read an existing Sherpa plasma model

**NOTE:** The following API requires the compiling of the XSPEC bridge
when installing.

```bash
export SHERPA_PLASMA_BUILD_XSPEC_BRIDGE=1
python -m pip install '.[plasma]'
```

The primary usage of this module is for querying the AtomDB database
for line information based on the best fit plasma model to an X-ray
spectrum. `sherpa_plasma` provides functions for reading and capturing
plasma models from a Sherpa session. Both equilibrium and non-equilibrium
models are supported along with multi-temperature, complex model
expressions (e.g., `phabs*bvapec + bvvapec + bapec`).

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

Compound expressions and shared-grid wrappers are traversed in expression
order; each unique component is returned once. Other components, such as
absorption and continua, are ignored. An expression without a supported APEC
component raises an error.

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
applied. 

**NOTE:** `Sedov` and other XSPEC models without a direct PyAtomDB mapping are
not supported.

The same `brightest=` model-ranking option in `find_lines` and the component
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

## Comparing Plasma Models

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

## Abundances

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