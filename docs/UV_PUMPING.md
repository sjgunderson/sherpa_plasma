# UV pumping of He-like triplets

`wrap_triplet_pumping` adds an AtomDB-backed source-level correction to an existing Sherpa
`XSapec`, `XSvapec`, or `XSvvapec` component, with limited experimental
`XSwdem`, `XSvwdem`, `XSvvwdem`, their broadened counterparts, and
`XSmkcflow` support described below. It removes photons from the He-like forbidden line and
adds them to the two intercombination lines. The child model and its parameter
objects remain the same; continuum, resonance lines, satellites, and other
components are not intentionally changed. The correction is applied before
the instrument response.

## Requirements and example

Use Sherpa with XSPEC 12.14.0k, Astropy, and an absolute `APECROOT` for
the active AtomDB line and continuum files. Bundled calibrations are selected
by the SHA256 hashes of the AtomDB 3.0.9 and 3.1.3 reference files. For other
installations, the wrapper derives the triplet calibration from a matching
local AtomDB filemap and its LV/LA atomic files; see the example below. The
wrapper accepts any abundance table selected in XSPEC, including built-in
tables and custom files or vectors. Apart from the trace-abundance settings
described below, other APEC XSET options, including broadening and
alternate interpolation settings, and the `LINECRITLEVEL` XSET are rejected.
Install the package into the same environment as Sherpa:

```python
from sherpa.astro import xspec
from sherpa.astro import ui
from sherpa_plasma import AtomDB, wrap_triplet_pumping

# Substitute the absolute root of your matching *_line.fits and *_coco.fits.
xspec.set_xsxset("APECROOT", "/absolute/path/to/apec_v3.1.3")
xspec.set_xsabund("angr")

plasma = xspec.XSvvapec("plasma")
plasma.kT = 0.43  # keV; the wrapper follows the full native model temperature rule
pumped = wrap_triplet_pumping(plasma, ["Na"])
pumped.p_Na = 1.0
print(pumped.temperature_coverage())
print(pumped.diagnostics())

# After loading a dataset and response in Sherpa:
ui.set_source(pumped)
```

For an installation without a bundled calibration, pass an `AtomDB`
connection identifying that installation, including versioned emissivity
file names when necessary:

```python
db = AtomDB("/absolute/path/to/atomdb",
            linefile="apec_vX.Y.Z_line.fits",
            cocofile="apec_vX.Y.Z_coco.fits",
            filemap="filemap")
xspec.set_xsxset("APECROOT", "/absolute/path/to/atomdb/apec_vX.Y.Z")
plasma = xspec.XSwdem("dem")
pumped = wrap_triplet_pumping(plasma, ["Na"], atomdb=db)
```

Alternatively, with `atomdb` omitted, an unmatched file pair is resolved
through `$ATOMDB`, using `apec_line.fits`, `apec_coco.fits`, and `filemap`.
The provider's line and continuum files must match the active XSPEC files
by SHA256. The filemap and atomic data must belong to that same emissivity
release; matching the emissivity hashes alone cannot establish that fact.
The files must be present locally; no downloads occur. The `VERSION` file,
if present, supplies the diagnostic version label, not a compatibility check.

The parameter `p_Na` is nonnegative, dimensionless, and may be fitted or
frozen like other Sherpa parameters. Its initial value is zero; at zero the
wrapper returns the unmodified child spectrum. Choose other ions by symbol or
atomic number, for example `wrap_triplet_pumping(plasma, ["O", "Ne", "Na"])`.
Each selected ion has its own `p_<symbol>` parameter.

### Abundance tables

Select the composition before constructing the wrapper. For example,
`xspec.set_xsabund("lodd")` selects a built-in table, while
`xspec.set_xsabund("/absolute/path/to/abundances.dat")` loads a custom file.
Custom vectors can also be supplied directly:

```python
composition = xspec.get_xsabundances("angr")
composition["Si"] *= 2.0
xspec.set_xsabundances(composition)
pumped = wrap_triplet_pumping(plasma, ["Si"])
```

The unbroadened correction converts reference APEC emissivities by the selected
element's active abundance divided by its `angr` abundance. Native broadened
profiles already include that conversion inside XSPEC, so it is applied once.
The component's fitted element abundance and normalization are then applied
with the existing scalar, variable-element, and trace-element conventions.
Values must be finite and nonnegative. A zero table abundance produces no
correction for that element, and diagnostics report no triplet emission.

If the table changes, reconstruct the wrapper. Custom-file/vector changes are
detected from their numerical values even when XSPEC continues to call the
table `file`. Reconstructing also clears the child's cached spectrum so the
new composition is evaluated. Named-table checks use the table name; they do
not read the full abundance vector on every evaluation. Cooling-flow weights
are captured from the model under the selected composition, rather than
rescaling weights from a different table.

The supported ions for the bundled reference datasets are C V, N VI, O VII,
Ne IX, Na X, Mg XI, Al XII, Si XIII, P XIV, S XV, Cl XVI, Ar XVII, K XVIII,
Ca XIX, Ti XXI, Cr XXIII, Mn XXIV, Fe XXV, and Ni XXVII. Other ions are
rejected for those datasets. For other installations, availability is
determined from the selected ion's LV/LA data and transition identities. Broadened APEC variants and arbitrary plasma models are not supported.

### Experimental wDEM support

`wrap_triplet_pumping` also accepts `XSwdem`, `XSvwdem`, `XSvvwdem`,
`XSbwdem`, `XSbvwdem`, and `XSbvvwdem` components under a
strict experimental configuration: available ions from the active AtomDB
installation, any active XSPEC abundance table, `switch=2`, `nH=1`, and no APEC
XSETs other than `APECROOT`. The optional XSPEC bridge must be built as
described in [TESTING.md](TESTING.md). The component's scalar metal abundance
also scales sodium, unlike the default trace-element treatment in `XSapec`.
Other trace elements not exposed by `XSwdem` retain solar abundance.
`XSvwdem` uses the selected ion's individual abundance where exposed and
solar abundance for unexposed trace elements; `XSvvwdem` uses individual
abundances for all supported ions. The corresponding broadened variants use
the same abundance conventions.

The wrapper captures the native temperature weights from the same XSPEC
evaluation, integrates the unpumped triplet fluxes across temperature and
applies `R_integrated = R0_integrated / (1 + p)` to those integrated fluxes.
For every positive-weight DEM bin, the forbidden and two intercombination
transitions are evaluated independently and their photon emissivities summed
with the original weights. A transition absent from a native temperature's line
list contributes zero at that endpoint, just as in XSPEC. Other transitions at
that temperature still contribute. Interpolation is linear in temperature under
the supported default XSPEC settings. Values beyond the native APEC grid use
the nearest endpoint, following the underlying model. The wrapper never cuts
or renormalizes the DEM to a complete-triplet interval.

This applies at every temperature, including `Tmax=1`, `beta=.1` for Si or Na
and DEMs spanning both low- and high-temperature tails. There is no coverage
error budget. The former `coverage_tolerance` keyword remains accepted for
compatibility but has no effect.

A component producing no triplet photons is unchanged. If the integrated
forbidden flux is zero, there are no photons to transfer. If the integrated
intercombination flux is zero, `R0` is infinite and finite dimensionless `p`
does not define a finite suppression target; the component is unchanged and
`diagnostics()` reports `ratio_status="no_intercombination_emission"`. A wholly
absent triplet has `R0=None`, `R=None`, and `ratio_status="no_triplet_emission"`.
For emitting triplets, the status is `"defined"` and the usual integrated ratio
law applies. The two intercombination branches need not both be nonzero.

At zero pumping the child spectrum is unchanged. For DEM models, `diagnostics()` reports
`R0_integrated` and `R_integrated`, the ratios of the temperature-integrated
triplet fluxes, and the contributing temperature range and count. It evaluates
the current model on a fixed diagnostic energy grid to capture weights; these
ratios do not represent a single-temperature critical rate. Applying the same
`p` separately at every temperature is a different model and generally would
not produce this integrated ratio.

This limited `XSwdem` path was compared against independently isolated triplet
lines for every bundled 3.0.9 ion at one parameter set, with additional parameter
sets for sodium and oxygen. Selected 3.1.3 cases were also compared.
Variable-abundance models were compared for
sodium and phosphorus, including an unexposed trace element in `XSvwdem`.
For broadened variants, the optional native profile bridge uses the same
XSPEC APEC implementation as Sherpa to project isolated temperature-weighted
lines, including thermal and velocity broadening. It scales each forbidden
and intercombination profile by its integrated transfer; the profile tails
outside the requested energy grid are not renormalized. The bridge verifies
that it uses the same XSPEC library as Sherpa. When photons are transferred,
an intercombination branch with zero integrated flux is populated using the
forbidden donor's temperature distribution at that branch's energy. Existing
nonzero branches retain their own normalized native profiles. Rebuild the
optional profile bridge after updating this code: the new `project_transfer`
entry point is required for full-DEM broadened support. Initializing a
new ion loads native APEC data, which can be expensive; up to four native
ion contexts are cached. At the final native temperature-grid endpoint,
the bridge uses the immediately lower representable temperature to avoid an
XSPEC boundary-search error.
Broadened comparisons cover all three variants for sodium at zero and
nonzero velocities, including truncated energy grids, plus selected oxygen,
phosphorus, and iron cases.
These historical comparisons do not establish accuracy for every installation,
abundance table, DEM family, or broadened APEC variant. The expanded abundance
regressions described in [TESTING.md](TESTING.md) require a native XSPEC run.
XSPEC global
metadata is shared: avoid concurrent XSPEC evaluations while the bridge
captures temperature weights.

### Experimental XSmkcflow support

The wrapper also accepts unbroadened `XSmkcflow` for the available ions, with
the active AtomDB files, any active XSPEC abundance table, `switch=2`, positive
redshift, and no APEC XSETs other than `APECROOT`. It requires the optional
XSPEC bridge and the same full-DEM integration and concurrency precautions
as `XSwdem`. The wrapper detects XSPEC cosmology changes and must be
reconstructed afterward.

The scalar abundance applies to the main elements and sodium; other trace
elements remain at solar abundance. The correction uses the original native cooling-flow temperature weights,
including their cooling-function and cosmological scaling. Validation compared
every calibrated ion at a subsolar abundance, plus three sodium parameter sets
including sub- and supersolar abundances, against independently
isolated single-temperature APEC lines
summed with those fixed weights; each unmodified weighted APEC spectrum was
also checked against the native cooling-flow spectrum. Removing a line directly
from the cooling-flow database would change its cooling function and is not
an equivalent reference. This does not validate other
cooling-flow variants, or broadened profiles.

For `XSapec`, the fitted metal abundance applies to the main elements; for
`XSvapec`, their individual abundances apply. For trace elements not exposed
by those models, the default abundance is 1 relative to solar. The supported
`APEC_TRACE_ABUND` settings are a nonnegative numeric value or the exact
string `Fe`; the latter tracks the fitted Fe abundance (or the common metal
abundance in `XSapec`). Other values are rejected. For `XSvvapec`, use its
individual element parameters and leave `APEC_TRACE_ABUND` unset.

Single-temperature components use the same full-grid rule. `temperature_coverage()`
reports the native database intervals, `policy="full_dem"`, and
`outside_native_grid="nearest_endpoint"`; these describe the underlying table,
not restrictions on the component's temperatures. The wrapper leaves the child's
temperature bounds unchanged. DEM diagnostics include all positive-weight
bins in their temperature range and count, and report the three integrated
intrinsic photon emissivities in `intrinsic_triplet_fluxes`. These emissivities
include the active table's solar abundance scale, before the fitted element
abundance and normalization. `abundance_scale_from_angr` reports that scale.

Use one wrapper per plasma component in a sum. If physically justified,
parameters can be linked with Sherpa parameter expressions. Equal `p` values
for different ions do not in general mean equal physical rates or densities.

## Suppression convention and assumptions

For each ion, `p_Z` is a nonnegative dimensionless suppression strength.
The wrapper does not calculate or require a critical rate. The user may
interpret `p_Z` as `phi/phi_c` or `n/n_c` when the corresponding physical
assumptions are appropriate. For single-temperature models, `diagnostics()`
reports `p`, the unpumped ratio `R0`, suppressed ratio `R`, file hashes,
and omitted decay fractions. For DEM models the reported ratios are based
on the temperature-integrated triplet.

The model uses `R = R0 / (1 + p)` and conserves the intrinsic
forbidden-plus-intercombination photon flux. The split between the two
intercombination lines uses version-matched AtomDB radiative data and
statistical weights `(1, 3, 5)/9`, including cascades between triplet P
levels. The ratio alone does not determine that split; in particular this
prescription does not establish that density-driven redistribution follows
the same branching. The wrapper is not a radiation-field calculation or a
general collisional level-population solver. For wDEM, the child's `nH=1`
restriction remains in force: a density interpretation of `p` does not change
the density used by the underlying XSPEC plasma model.

**Migration:** Previously, the same parameter name meant
`phi/phi_c(T_ref)` and was converted to a physical rate before evaluation.
Existing nonzero fitted values generally cannot be reused unchanged. For a
single-temperature model, an equivalent new value is the old physical `phi`
divided by the old critical rate at that temperature. A DEM conversion
depends on the old model and its integrated triplet and is not a universal
temperature-independent rescaling. Save any needed old diagnostics before
upgrading.

Other radiative channels are omitted only if their rate is at most 1e-12 of
the modeled decay rate of the originating P level. In the packaged records
the maximum such fraction is less than 9.85e-13, and the smallest effective
branching is greater than 0.077. Since a downward cascade can visit at most
three P levels, the probability of an omitted channel per excitation is
less than 3e-12; relative to that minimum effective branching this is less
than 4e-11. The exact triplet-conservation model therefore neglects these
small channels, and this bound is not a statement about other atomic or
astrophysical uncertainties.

## Practical limitations

- Pumped spectra require contiguous, increasing source-energy bins. Nonuniform
  widths are supported. For unbroadened models, energies too close to a bin
  edge are rejected because exact XSPEC edge behavior is not established.
- Nonzero pumping needs matching atomic transition identities and branching
  data. Native table entries may be absent at any temperature; this contributes
  zero model emissivity for that transition. Unsupported settings and unmatched
  datasets still fail explicitly.
- Treat APEC files as immutable while XSPEC is running. If they change, restart
  the Python/XSPEC process and reconstruct the wrapper. The wrapper detects
  changes after construction, but cannot determine whether XSPEC had already
  cached older contents before it was constructed.
- Changes to the relevant XSPEC global settings, abundance table, or numerical
  custom abundance vector require reconstructing the wrapper.
- Packaged calibrations apply only to their exact file hashes. Other file
  pairs require the matching installation's filemap and atomic files; data
  that fail the atomic-identity or branching checks are rejected.
- Changing any filemap, LV, or LA file used for an installed calibration
  requires reconstructing the wrapper.
- Sherpa's binary `save`/`restore` round-trip has been tested in the same
  environment. `save_all` writes a text script but does not reconstruct this
  custom wrapper; recreate it with `wrap_triplet_pumping` when using that script.
- Broadened APEC variants such as `XSbvvapec` are not currently supported.

See [TESTING.md](TESTING.md) for reproducible isolated-line comparisons and
benchmarks. The validation checks the supported reference files against
compiled XSPEC; it does not establish accuracy for untested software versions,
settings, broadened profiles, or atomic datasets not covered by those tests.

The calibration files were generated from the matching AtomDB release
filemaps and LV/LA atomic files. See the
[AtomDB database documentation](https://www.atomdb.org/Features/database.php), the
[XSPEC APEC manual](https://heasarc.gsfc.nasa.gov/docs/software/xspec/manual/XSmodelApec.html),
and [Porquet et al. (2001)](https://arxiv.org/abs/astro-ph/0107329) for
background on He-like triplet ratios and UV pumping.

