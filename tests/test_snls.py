import numpy as np
import pytest
from typer.testing import CliRunner

from gcphotom.cli import app
from gcphotom.surveys.snls import (
    _add_fields,
    _parse_radius_columns,
    build_contamination_mask,
    default_output_path,
    filter_to_reference,
    parse_snls_catalog,
    process_single,
    write_catalog,
)


def _make_forced_catalog(n_src=10, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    radii = [1.0, 2.0, 3.0]
    dt = [("bindex", "<i8"), ("windowed", "?"), ("saturated", "?")]
    for prefix in ("apfl", "apvar", "apother"):
        for r in radii:
            dt.append((f"{prefix}_{r:.2f}", "<f8"))
    arr = np.empty(n_src, dtype=dt)
    arr["bindex"] = rng.integers(0, 5, size=n_src)
    arr["windowed"] = False
    arr["saturated"] = False
    for r in radii:
        arr[f"apfl_{r:.2f}"] = rng.uniform(10, 100, size=n_src)
        arr[f"apvar_{r:.2f}"] = rng.uniform(1, 10, size=n_src)
        arr[f"apother_{r:.2f}"] = rng.uniform(0, 5, size=n_src)
    return arr, radii


def _make_ref_catalog(n_ref=5, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    dt = [
        ("index", "<i8"),
        ("star", "?"),
        ("star_g", "?"),
        ("flux_g", "<f8"),
    ]
    arr = np.empty(n_ref, dtype=dt)
    arr["index"] = np.arange(n_ref)
    for col in ("star", "star_g"):
        arr[col] = rng.choice([True, False], size=n_ref)
    arr["flux_g"] = rng.uniform(5000, 20000, size=n_ref)
    return arr


class TestParseRadiusColumns:
    def test_basic(self):
        cat, _ = _make_forced_catalog()
        radii = _parse_radius_columns(cat)
        assert radii == [1.0, 2.0, 3.0]

    def test_empty(self):
        cat = np.empty(0, dtype=[("x", "<f8")])
        assert _parse_radius_columns(cat) == []


class TestParseSnlsCatalog:
    def test_shapes_and_keys(self, tmp_path):
        cat, radii = _make_forced_catalog(n_src=10)
        path = tmp_path / "test.npy"
        np.save(str(path), cat)
        gc_result, meta = parse_snls_catalog(path)
        assert set(gc_result) == {
            "radius",
            "flux",
            "flux_clean",
            "background_var",
            "contamination",
        }
        assert list(gc_result["radius"]) == radii
        assert gc_result["flux"].shape == (10, 3)
        assert gc_result["flux_clean"].shape == (10, 3)
        assert gc_result["background_var"].shape == (10, 3)
        assert gc_result["contamination"].shape == (10, 3)
        # flux_clean is now a copy of flux (no subtraction)
        assert np.allclose(gc_result["flux_clean"], gc_result["flux"])
        # contamination is zeroed out (provided separately in meta)
        assert np.allclose(gc_result["contamination"], 0)
        # meta contains the original cat and contamination/bad arrays
        assert "cat" in meta
        assert np.array_equal(meta["cat"], cat)
        assert "contamination" in meta
        assert meta["contamination"].shape == (10, 3)
        assert "bad" in meta
        assert meta["bad"].shape == (10, 3)

    def test_single_source(self, tmp_path):
        cat, _ = _make_forced_catalog(n_src=1)
        path = tmp_path / "single.npy"
        np.save(str(path), cat)
        gc_result, _ = parse_snls_catalog(path)
        assert gc_result["flux"].shape == (1, 3)

    def test_no_apfl_columns(self, tmp_path):
        cat = np.empty(5, dtype=[("x", "<f8"), ("y", "<f8")])
        path = tmp_path / "empty.npy"
        np.save(str(path), cat)
        gc_result, _ = parse_snls_catalog(path)
        assert gc_result["flux"].shape == (5, 0)


class TestFilterToReference:
    def test_basic_filter(self):
        rng = np.random.default_rng(99)
        ref = _make_ref_catalog(n_ref=5, rng=rng)
        ref["star"] = [True, True, False, True, True]
        ref["star_g"] = [True, True, False, True, True]
        ref["flux_g"] = [5000, 15000, 20000, 12000, 8000]
        cat, _ = _make_forced_catalog(n_src=8, rng=rng)
        # Force specific bindex values
        cat["bindex"] = [0, 1, 2, 3, 4, 0, 1, 2]
        cat["windowed"] = [False, False, False, False, False, True, False, False]
        cat["saturated"] = [False, False, False, False, False, False, False, False]
        mask = filter_to_reference(cat, ref, band="g", min_flux=10000.0)
        # Source 0: star_g=True, flux_g=5000 → False (min_flux)
        # Source 1: star_g=True, flux_g=15000 → True
        # Source 2: star_g=False → False
        # Source 3: star_g=True, flux_g=12000 → True
        # Source 4: star_g=True, flux_g=8000 → False (min_flux)
        # Source 5: same as 0, windowed=True → False
        # Source 6: same as 1 → True
        # Source 7: same as 2 → False
        expected = [False, True, False, True, False, False, True, False]
        assert list(mask) == expected

    def test_fallback_star_flag(self):
        ref = np.empty(3, dtype=[("index", "<i8"), ("star", "?"), ("flux_g", "<f8")])
        ref["index"] = [0, 1, 2]
        ref["star"] = [True, False, True]
        ref["flux_g"] = [20000, 20000, 5000]
        cat, _ = _make_forced_catalog(n_src=3)
        cat["bindex"] = [0, 1, 2]
        mask = filter_to_reference(cat, ref, band="g", min_flux=10000.0)
        assert list(mask) == [True, False, False]

    def test_exclude_windowed(self):
        ref = np.empty(2, dtype=[("index", "<i8"), ("star", "?"), ("flux_g", "<f8")])
        ref["index"] = [0, 1]
        ref["star"] = [True, True]
        ref["flux_g"] = [20000, 20000]
        cat, _ = _make_forced_catalog(n_src=2)
        cat["bindex"] = [0, 1]
        cat["windowed"] = [False, True]
        mask = filter_to_reference(cat, ref, band="g", exclude_windowed=True)
        assert list(mask) == [True, False]

    def test_exclude_saturated(self):
        ref = np.empty(2, dtype=[("index", "<i8"), ("star", "?"), ("flux_g", "<f8")])
        ref["index"] = [0, 1]
        ref["star"] = [True, True]
        ref["flux_g"] = [20000, 20000]
        cat, _ = _make_forced_catalog(n_src=2)
        cat["bindex"] = [0, 1]
        cat["saturated"] = [False, True]
        mask = filter_to_reference(cat, ref, band="g", exclude_saturated=True)
        assert list(mask) == [True, False]

    def test_no_match_returns_all_false(self):
        ref = np.empty(2, dtype=[("index", "<i8"), ("star", "?"), ("flux_g", "<f8")])
        ref["index"] = [10, 20]
        ref["star"] = [True, True]
        ref["flux_g"] = [20000, 20000]
        cat, _ = _make_forced_catalog(n_src=3)
        cat["bindex"] = [0, 1, 2]
        mask = filter_to_reference(cat, ref, band="g")
        assert list(mask) == [False, False, False]


class TestWriteCatalog:
    def test_appends_columns(self, tmp_path):
        rng = np.random.default_rng(42)
        cat, _ = _make_forced_catalog(n_src=5, rng=rng)
        result = {
            "cat": cat,
            "bf": {"gamma": 3.0, "alpha": 3.5, "flux": np.ones(5), "back": np.zeros(5)},
            "fitter": None,  # would need a real fitter; tested via integration
        }
        out_path = tmp_path / "result.npy"
        with pytest.raises(AttributeError):  # fitter is None
            write_catalog(result, out_path)

    def test_default_output_path(self):
        assert (
            str(default_output_path("catalog_forced_D1_g_01.npy"))
            == "catalog_mophot_D1_g_01.npy"
        )
        assert (
            str(default_output_path("/path/to/catalog_forced.npy"))
            == "/path/to/catalog_mophot.npy"
        )


class TestBuildContaminationMask:
    def test_clean_apother(self):
        meta = {
            "contamination": np.zeros((5, 3)),
            "bad": np.zeros((5, 3)),
        }
        mask = build_contamination_mask(meta)
        assert mask.shape == (3, 5)  # (n_radii, n_sources) per Fitter convention
        assert mask.all()

    def test_some_contamination(self):
        # cum_cont shape (n_sources, n_radii)
        cum_cont = np.array(
            [
                [0, 1, 1],  # source 0: cont at radii 1,2
                [0, 0, 2],  # source 1: cont at radius 2
                [0, 0, 0],  # source 2: no cont
            ]
        )
        meta = {
            "contamination": cum_cont,
            "bad": np.zeros_like(cum_cont),
        }
        mask = build_contamination_mask(meta)  # returns (n_radii, n_sources)
        # Annular cont per source:
        #   src 0: [0, 1, 0] → good at bins 0,2
        #   src 1: [0, 0, 2] → good at bins 0,1
        #   src 2: [0, 0, 0] → all good
        # Transposed to (n_radii, n_sources):
        #   rad 0: [T, T, T]
        #   rad 1: [F, T, T]
        #   rad 2: [T, F, T]
        expected = np.array(
            [
                [True, True, True],
                [False, True, True],
                [True, False, True],
            ]
        )
        assert np.array_equal(mask, expected)

    def test_some_bad_pixels(self):
        # cum_bad shape (n_sources, n_radii)
        cum_bad = np.array(
            [
                [0, 1],  # source 0: bad at radius 1
                [0, 0],  # source 1: no bad
            ]
        )
        meta = {
            "contamination": np.zeros_like(cum_bad),
            "bad": cum_bad,
        }
        mask = build_contamination_mask(meta)
        # Annular bad per source:
        #   src 0: [0, 1] → good at bin 0
        #   src 1: [0, 0] → all good
        # Transposed to (n_radii, n_sources):
        #   rad 0: [T, T]
        #   rad 1: [F, T]
        expected = np.array(
            [
                [True, True],
                [False, True],
            ]
        )
        assert np.array_equal(mask, expected)

    def test_no_apbad_in_catalog(self, tmp_path):
        """When the catalog has no apbad columns, bad array should be all zeros."""
        cat, _ = _make_forced_catalog(n_src=3)
        path = tmp_path / "test.npy"
        np.save(str(path), cat)
        _, meta = parse_snls_catalog(path)
        assert meta["bad"].shape == (3, 3)
        assert meta["bad"].sum() == 0

    def test_with_apbad_columns(self, tmp_path):
        """When the catalog has apbad columns, they should be read into meta."""
        radii = [1.0, 2.0]
        dt = [("bindex", "<i8"), ("windowed", "?"), ("saturated", "?")]
        for prefix in ("apfl", "apvar", "apother", "apbad"):
            for r in radii:
                dt.append((f"{prefix}_{r:.2f}", "<f8"))
        cat = np.empty(3, dtype=dt)
        cat["bindex"] = [0, 1, 2]
        cat["windowed"] = False
        cat["saturated"] = False
        for r in radii:
            cat[f"apfl_{r:.2f}"] = [100, 200, 300]
            cat[f"apvar_{r:.2f}"] = [10, 10, 10]
            cat[f"apother_{r:.2f}"] = [0, 0, 0]
            cat[f"apbad_{r:.2f}"] = [0, 1, 2]
        path = tmp_path / "with_apbad.npy"
        np.save(str(path), cat)
        _, meta = parse_snls_catalog(path)
        assert meta["bad"].shape == (3, 2)
        assert meta["bad"][0, 0] == 0
        assert meta["bad"][1, 0] == 1


class TestAddFields:
    def test_adds_fields_correctly(self):
        base = np.array([(1, 2.0), (3, 4.0)], dtype=[("a", "<i8"), ("b", "<f8")])
        result = _add_fields(base, c=[10.0, 20.0], d=[100.0, 200.0])
        assert "c" in result.dtype.names
        assert "d" in result.dtype.names
        assert list(result["c"]) == [10.0, 20.0]
        assert list(result["d"]) == [100.0, 200.0]
        assert list(result["a"]) == [1, 3]
        assert list(result["b"]) == [2.0, 4.0]

    def test_preserves_original_fields(self):
        base = np.array([(1, 2.0)], dtype=[("a", "<i8"), ("b", "<f8")])
        result = _add_fields(base, c=[3.0])
        assert result["a"] == 1
        assert result["b"] == 2.0


class TestCLI:
    def test_snls_process_help(self):
        runner = CliRunner()
        result = runner.invoke(app, ["snls", "process", "--help"])
        assert result.exit_code == 0
        assert "Fit growth curves" in result.stdout

    def test_snls_match_help(self):
        runner = CliRunner()
        result = runner.invoke(app, ["snls", "match", "--help"])
        assert result.exit_code == 0
        assert "matching statistics" in result.stdout

    def test_snls_process_no_args(self):
        runner = CliRunner()
        result = runner.invoke(app, ["snls", "process"])
        assert result.exit_code != 0

    def test_snls_process_no_match(self, tmp_path):
        ref = _make_ref_catalog(n_ref=2)
        ref["star_g"] = [False, False]
        ref_path = tmp_path / "ref.npy"
        np.save(str(ref_path), ref)
        cat, _ = _make_forced_catalog(n_src=5, rng=np.random.default_rng(42))
        cat["bindex"] = [0, 0, 0, 0, 0]
        cat_path = tmp_path / "cat.npy"
        np.save(str(cat_path), cat)
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "snls",
                "process",
                str(cat_path),
                "--reference",
                str(ref_path),
            ],
        )
        assert result.exit_code == 0
        assert "No matched sources" in result.stdout

    def test_snls_match_shows_counts(self, tmp_path):
        ref = _make_ref_catalog(n_ref=2)
        ref["star_g"] = [True, True]
        ref["flux_g"] = [20000, 20000]
        ref_path = tmp_path / "ref.npy"
        np.save(str(ref_path), ref)
        cat, _ = _make_forced_catalog(n_src=5)
        cat["bindex"] = [0, 0, 0, 0, 0]
        cat_path = tmp_path / "cat.npy"
        np.save(str(cat_path), cat)
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "snls",
                "match",
                str(cat_path),
                "--reference",
                str(ref_path),
            ],
        )
        assert result.exit_code == 0
        assert "Total" in result.stdout
        assert "Matched" in result.stdout

    def test_snls_match_dump(self, tmp_path):
        ref = _make_ref_catalog(n_ref=2)
        ref["star_g"] = [True, True]
        ref["flux_g"] = [20000, 20000]
        ref_path = tmp_path / "ref.npy"
        np.save(str(ref_path), ref)
        cat, _ = _make_forced_catalog(n_src=5)
        cat["bindex"] = [0, 0, 0, 0, 0]
        cat_path = tmp_path / "cat.npy"
        np.save(str(cat_path), cat)
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "snls",
                "match",
                str(cat_path),
                "--reference",
                str(ref_path),
                "--dump",
            ],
        )
        assert result.exit_code == 0
        assert "100.0%" in result.stdout
        out_path = tmp_path / "cat_matched.npy"
        assert out_path.exists()
        dumped = np.load(str(out_path))
        assert len(dumped) == 5
        np.testing.assert_array_equal(cat["bindex"], dumped["bindex"])

    def test_snls_match_dump_dir(self, tmp_path):
        ref = _make_ref_catalog(n_ref=2)
        ref["star_g"] = [True, True]
        ref["flux_g"] = [20000, 20000]
        ref_path = tmp_path / "ref.npy"
        np.save(str(ref_path), ref)
        cat, _ = _make_forced_catalog(n_src=5)
        cat["bindex"] = [0, 0, 0, 0, 0]
        cat_path = tmp_path / "cat.npy"
        np.save(str(cat_path), cat)
        out_dir = tmp_path / "output"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "snls",
                "match",
                str(cat_path),
                "--reference",
                str(ref_path),
                "--dump-dir",
                str(out_dir),
            ],
        )
        assert result.exit_code == 0
        out_path = out_dir / "cat_matched.npy"
        assert out_path.exists()
        dumped = np.load(str(out_path))
        assert len(dumped) == 5

    def test_snls_process_glob_no_match(self, tmp_path):
        """Glob pattern that matches nothing should exit with error."""
        runner = CliRunner()
        ref_path = tmp_path / "ref.npy"
        np.save(str(ref_path), _make_ref_catalog(n_ref=1))
        result = runner.invoke(
            app,
            [
                "snls",
                "process",
                str(tmp_path / "nonexistent_*.npy"),
                "--reference",
                str(ref_path),
            ],
        )
        assert result.exit_code != 0
        assert "No files matching" in result.stdout

    def test_snls_match_glob_no_match(self, tmp_path):
        """Glob pattern that matches nothing should exit with error."""
        runner = CliRunner()
        ref_path = tmp_path / "ref.npy"
        np.save(str(ref_path), _make_ref_catalog(n_ref=1))
        result = runner.invoke(
            app,
            [
                "snls",
                "match",
                str(tmp_path / "nonexistent_*.npy"),
                "--reference",
                str(ref_path),
            ],
        )
        assert result.exit_code != 0
        assert "No files matching" in result.stdout


class TestProcessSingle:
    """Integration-level tests for process_single that exercise the full fitting pipeline."""

    def test_process_single_round_trip(self, tmp_path):
        rng = np.random.default_rng(42)
        # Forced catalog: 10 sources, 4 radii
        radii = [1.0, 2.0, 4.0, 8.0]
        n_src = 10
        dt = [("bindex", "<i8"), ("windowed", "?"), ("saturated", "?")]
        for prefix in ("apfl", "apvar", "apother"):
            for r in radii:
                dt.append((f"{prefix}_{r:.2f}", "<f8"))
        cat = np.empty(n_src, dtype=dt)
        cat["bindex"] = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
        cat["windowed"] = False
        cat["saturated"] = False
        # Create a Moffat-like flux pattern so the fitter converges
        base_flux = 500.0 * (1 - 1 / (1 + np.array(radii) ** 2 / 9))
        for i in range(n_src):
            scale = 0.5 + 0.1 * i
            for j, r in enumerate(radii):
                val = scale * base_flux[j] + rng.normal(0, 1)
                cat[f"apfl_{r:.2f}"][i] = max(val, 0)
                cat[f"apvar_{r:.2f}"][i] = abs(rng.normal(100, 20))
                cat[f"apother_{r:.2f}"][i] = 0.0

        cat_path = tmp_path / "catalog_forced_D1_g_test.npy"
        np.save(str(cat_path), cat)

        # Reference catalog: 4 sources, all stars with flux > threshold
        ref = np.empty(
            4,
            dtype=[("index", "<i8"), ("star", "?"), ("star_g", "?"), ("flux_g", "<f8")],
        )
        ref["index"] = [0, 1, 2, 3]
        ref["star"] = True
        ref["star_g"] = True
        ref["flux_g"] = [15000, 15000, 15000, 15000]
        ref_path = tmp_path / "ref.npy"
        np.save(str(ref_path), ref)

        result = process_single(
            cat_path,
            ref,
            band="g",
            min_flux=10000.0,
            learning_rate=1e-2,
            niter=50,
        )
        assert result is not None
        assert result["n_initial"] == 10
        assert result["n_selected"] == 10  # all pass filter
        assert "bf" in result
        assert "fitter" in result
        assert "gamma" in result["bf"]
        assert "alpha" in result["bf"]

        # Write and verify output
        out_path = tmp_path / "catalog_mophot_D1_g_test.npy"
        write_catalog(result, out_path)
        assert out_path.exists()
        out_cat = np.load(str(out_path))
        assert "mflux" in out_cat.dtype.names
        assert "mback" in out_cat.dtype.names
        assert "mgoods" in out_cat.dtype.names
        assert "mchi2" in out_cat.dtype.names
        # All original columns should be present
        for name in cat.dtype.names:
            assert name in out_cat.dtype.names

        # Result size matches selected sources
        assert len(out_cat) == result["n_selected"]

    def test_process_single_nothing_selected(self, tmp_path):
        rng = np.random.default_rng(42)
        cat, _ = _make_forced_catalog(n_src=5, rng=rng)
        cat["bindex"] = [0, 0, 0, 0, 0]
        cat_path = tmp_path / "cat.npy"
        np.save(str(cat_path), cat)

        ref = np.empty(
            2,
            dtype=[("index", "<i8"), ("star", "?"), ("star_g", "?"), ("flux_g", "<f8")],
        )
        ref["index"] = [0, 1]
        ref["star"] = False
        ref["star_g"] = False
        ref["flux_g"] = [5000, 5000]

        result = process_single(cat_path, ref, band="g", min_flux=10000.0)
        assert result is None

    def test_process_single_with_output_via_cli(self, tmp_path):
        """End-to-end CLI test with real fitting."""
        rng = np.random.default_rng(42)
        radii = [1.0, 2.0, 4.0, 8.0]
        n_src = 6
        dt = [("bindex", "<i8"), ("windowed", "?"), ("saturated", "?")]
        for prefix in ("apfl", "apvar", "apother"):
            for r in radii:
                dt.append((f"{prefix}_{r:.2f}", "<f8"))
        cat = np.empty(n_src, dtype=dt)
        cat["bindex"] = [0, 1, 2, 0, 1, 2]
        cat["windowed"] = False
        cat["saturated"] = False
        base_flux = 500.0 * (1 - 1 / (1 + np.array(radii) ** 2 / 9))
        for i in range(n_src):
            scale = 0.5 + 0.2 * i
            for j, r in enumerate(radii):
                cat[f"apfl_{r:.2f}"][i] = max(scale * base_flux[j], 0)
                cat[f"apvar_{r:.2f}"][i] = 100.0
                cat[f"apother_{r:.2f}"][i] = 0.0

        cat_path = tmp_path / "catalog_forced_D1_g_test.npy"
        np.save(str(cat_path), cat)

        ref = np.empty(
            3,
            dtype=[("index", "<i8"), ("star", "?"), ("star_g", "?"), ("flux_g", "<f8")],
        )
        ref["index"] = [0, 1, 2]
        ref["star"] = True
        ref["star_g"] = True
        ref["flux_g"] = [20000, 20000, 20000]
        ref_path = tmp_path / "ref.npy"
        np.save(str(ref_path), ref)

        out_dir = tmp_path / "output"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "snls",
                "process",
                str(cat_path),
                "--reference",
                str(ref_path),
                "--output-dir",
                str(out_dir),
                "--learning-rate",
                "1e-2",
                "--niter",
                "50",
            ],
        )
        assert result.exit_code == 0, result.output
        # Check output file was created
        expected_out = out_dir / "catalog_mophot_D1_g_test.npy"
        assert expected_out.exists()
        out_cat = np.load(str(expected_out))
        assert "mflux" in out_cat.dtype.names
