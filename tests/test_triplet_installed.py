"""Triplet calibration and DEM behavior with non-bundled AtomDB files."""

import numpy as np
import pytest

pytest.importorskip("astropy.io.fits")
xspec = pytest.importorskip("sherpa.astro.xspec")

from sherpa_plasma import AtomDB, AtomDBTripletProvider, wrap_triplet_pumping
from sherpa_plasma._uv_atomic import _read_radiative_calibration
from sherpa_plasma._uv_pumping import _transfer_triplet_photons
from test_uv_atomic import _write_inputs
from test_uv_lines import _write_apec


@pytest.fixture
def installed_db(tmp_path):
    level, radiative = _write_inputs(tmp_path, cascade=100.)
    (tmp_path / "filemap").write_text(
        "2 8 7 $ATOMDB/levels.fits\n3 8 7 $ATOMDB/rates.fits\n",
        encoding="utf-8",
    )
    (tmp_path / "VERSION").write_text("custom-release\n", encoding="utf-8")
    rows = [(8, 7, 2, 1, 4.e-14, 22.1),
            (8, 7, 5, 1, 2.e-14, 21.3),
            (8, 7, 6, 1, 1.e-14, 21.7)]
    _write_apec(tmp_path / "apec_line.fits", [rows, rows])
    (tmp_path / "apec_coco.fits").write_bytes(b"test continuum")
    return AtomDB(tmp_path), level, radiative


@pytest.fixture
def mock_xspec(monkeypatch, installed_db):
    import sherpa_plasma._uv_model as model

    db, _, _ = installed_db
    monkeypatch.setattr(model, "_MANIFESTS", ())
    monkeypatch.setattr(model, "_root", lambda: db.root / "apec")
    monkeypatch.setattr(model, "_state", lambda: ("unchanged",))
    monkeypatch.setattr(xspec, "get_xsversion", lambda: "12.14.0k")
    monkeypatch.setattr(xspec, "get_xsabund", lambda: "angr")
    monkeypatch.setattr(xspec, "get_xsxset", lambda: {})
    monkeypatch.setattr(xspec, "get_xsabundances", lambda *args: {"O": 1.})
    monkeypatch.setattr(
        model, "_capture_cie_evaluation",
        lambda child, kind, trial, lo, hi, bridge, path:
            (np.ones_like(lo), np.array([.22, .28]), np.array([.25, .75])),
    )
    return model


def test_provider_uses_filemap_and_preserves_cascade_calibration(installed_db):
    db, level, radiative = installed_db
    provider = AtomDBTripletProvider(db)
    assert provider.atomic_files("O") == (level, radiative)
    assert provider.calibration("O") == _read_radiative_calibration(level, radiative)
    assert [line.upper for line in provider.line_ids("O")] == [2, 5, 6]


@pytest.mark.parametrize("kind", ["XSwdem", "XSvwdem", "XSvvwdem", "XSmkcflow"])
@pytest.mark.parametrize("automatic", [False, True])
def test_installed_dem_uses_its_calibration_and_conserves_photons(
        installed_db, mock_xspec, monkeypatch, kind, automatic):
    db, level, radiative = installed_db
    child = getattr(xspec, kind)("installed_dem")
    if kind == "XSmkcflow":
        child.Redshift = .02
    monkeypatch.setattr(type(child), "calc", lambda self, p, lo, hi: np.ones_like(lo))
    if automatic:
        monkeypatch.setenv("ATOMDB", str(db.root))
        wrapper = wrap_triplet_pumping(child, iter(["O"]))
    else:
        wrapper = wrap_triplet_pumping(child, "O", atomdb=db)
    edges = np.linspace(.4, .8, 1001)
    base = wrapper(edges[:-1], edges[1:])
    np.testing.assert_array_equal(base, np.ones_like(base))
    wrapper.p_O = 1.
    actual = wrapper(edges[:-1], edges[1:]) - base
    calibration = _read_radiative_calibration(level, radiative)
    assert wrapper._entries[0][0]["calibration"]["redistribution"] == calibration.redistribution
    assert np.count_nonzero(actual) == 3
    assert actual.sum() == pytest.approx(0., abs=1.e-12)
    f, i1, i2 = wrapper._entries[0][1].emissivities[0]
    transfer = _transfer_triplet_photons(f, i1, i2, 1.)
    norm = child.norm.val / (1 + child.Redshift.val)
    expected = 1.e14 * norm * transfer
    assert actual.min() == pytest.approx(-expected)
    assert np.sort(actual[actual > 0]) == pytest.approx(
        sorted(expected * np.asarray(calibration.redistribution)))
    diagnostic, = wrapper.diagnostics()
    assert diagnostic["atomdb_version"] == "custom-release"
    assert diagnostic["R_integrated"] == pytest.approx(diagnostic["R0_integrated"] / 2)


def test_provider_rejects_mismatched_apec_files(installed_db, mock_xspec):
    db, _, _ = installed_db
    different = AtomDB(db.root, cocofile="different.fits")
    different.cocofile.write_bytes(b"other continuum")
    with pytest.raises(ValueError, match="must match active APEC files"):
        wrap_triplet_pumping(xspec.XSwdem("different"), "O", atomdb=different)


def test_provider_rejects_missing_atomic_input(installed_db):
    db, _, radiative = installed_db
    radiative.unlink()
    with pytest.raises(FileNotFoundError, match="missing atomic file"):
        AtomDBTripletProvider(db).calibration("O")


def test_wrapper_detects_changed_atomic_input(installed_db, mock_xspec, monkeypatch):
    db, _, radiative = installed_db
    wrapper = wrap_triplet_pumping(xspec.XSwdem("changed"), "O", atomdb=db)
    with radiative.open("ab") as file:
        file.write(b"changed")
    with pytest.raises(RuntimeError, match="data files changed"):
        wrapper.temperature_coverage()


def test_wrapper_detects_filemap_symlink_retarget(installed_db, mock_xspec):
    db, _, _ = installed_db
    content = db.filemap.read_text(encoding="utf-8")
    first, second = db.root / "filemap-one", db.root / "filemap-two"
    first.write_text(content, encoding="utf-8")
    second.write_text(content, encoding="utf-8")
    db.filemap.unlink()
    db.filemap.symlink_to(first)
    wrapper = wrap_triplet_pumping(xspec.XSwdem("filemap_change"), "O", atomdb=db)
    db.filemap.unlink()
    db.filemap.symlink_to(second)
    with pytest.raises(RuntimeError, match="data files changed"):
        wrapper.temperature_coverage()


@pytest.mark.parametrize("kind", ["XSbwdem", "XSbvwdem", "XSbvvwdem"])
def test_installed_broadened_dem_passes_apec_identities_to_profile_bridge(
        installed_db, mock_xspec, monkeypatch, kind):
    from sherpa_plasma import _xspec_profiles

    db, _, _ = installed_db
    child = getattr(xspec, kind)("installed_broadened")
    monkeypatch.setattr(type(child), "calc", lambda self, p, lo, hi: np.ones_like(lo))
    wrapper = wrap_triplet_pumping(child, "O", atomdb=db)
    identities = []

    def load(*args):
        identities.append(args[-1])
        return object()

    def project(profiles, edges, temperatures, weights, redshift, velocity):
        result = np.zeros((3, len(edges) - 1))
        result[0, 100], result[1, 200], result[2, 300] = 4., 2., 1.
        return result

    monkeypatch.setattr(mock_xspec, "_load_broad_profiles", load)
    monkeypatch.setattr(_xspec_profiles, "project", project)
    edges = np.linspace(.4, .8, 1001)
    wrapper.p_O = 1.
    actual = wrapper(edges[:-1], edges[1:]) - 1.
    assert identities == [wrapper._data_identities[:2]]
    assert len(wrapper._data_identities) == 5
    assert np.count_nonzero(actual) == 3
    assert actual.sum() == pytest.approx(0., abs=1.e-6)
