# Non-Equilibrium Plasma Queries

Access to database queries for a non-equilibrium ionization (NEI) plasma is
can be called through PyAtomDB's `NEISession` and `PShockSession` options.
`sherpa_plasma` provides helper functions for querying these sessions to
extract line information.

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

The required parameters are explicit-parameter APIs from PyAtomDB's:
Ionization age (`tau`, `tau_lower`, `tau_upper`) is `n_e * t` in cm^-3 s.
Temperatures are in keV unless `teunit='K'` is specified. `init_pop` may be
`'ionizing'` (initially neutral), `'recombining'` (initially fully ionized),
or a scalar initial equilibrium temperature in the selected temperature unit.
The NEI line-emissivity API broadcasts temperature and ionization-age arrays;
the PShock API integrates over its lower and upper age limits.



Returned `Epsilon` values are photons $\mathrm{cm}^3$/s and include the selected
abundance baseline and multipliers. Missing local NEI tables, unsupported
initial populations, and invalid age bounds raise errors. Sherpa's bundled
eigenvector table is preferred; if it is unavailable, the local ion-balance data
is preloaded through the configured filemap. Missing entries are may be downloaded
through the PyAtomDB download scripts.

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
