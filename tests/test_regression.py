"""
Regression tests against the saved numerical baseline.

These tests catch unintended changes to the odds ratios, model means,
and covariance matrices introduced by bug fixes or refactoring.

Run after generating the baseline:
    pixi run python tests/save_baseline.py   # once, commit the output
    pixi run test-remote                     # on every change
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compass import model

BASELINES_DIR = Path(__file__).parent / "baselines"
SCALARS_PATH = BASELINES_DIR / "readme_example_scalars.json"
ARRAYS_PATH  = BASELINES_DIR / "readme_example_arrays.npz"

README_DATA = pd.DataFrame({
    "Main_ID":       ["HIP82545", "HIP82545", "HIP82545"],
    "final_uuid":    ["id0",       "id0",       "id0"],
    "date":          ["2016-04-24", "2018-04-24", "2021-06-04"],
    "dRA":           [-1390,        -1386,        -1394],
    "dRA_err":       [5,            5,            5],
    "dDEC":          [-990,         -989,         -995],
    "dDEC_err":      [5,            5,            5],
    "dRA_dDEC_corr": [0,            0,            0],
    "mag0":          [16,           16,           16],
    "mag0_err":      [1,            1,            1],
})


def baseline_exists():
    return SCALARS_PATH.exists() and ARRAYS_PATH.exists()


@pytest.fixture(scope="module")
def baseline():
    if not baseline_exists():
        pytest.skip("Baseline not generated yet — run tests/save_baseline.py first")
    with open(SCALARS_PATH) as f:
        scalars = json.load(f)
    arrays = np.load(ARRAYS_PATH)
    return scalars, arrays


@pytest.fixture(scope="module")
def survey_results():
    sv = model.Survey(README_DATA, "mag0")
    sv.set_fieldstar_models("ks_m_calc", "ks_m", cone_radius=0.1, binsize=50)
    sv.set_evaluated_fieldstar_models(sigma_cc_min=0, sigma_model_min=0)
    return sv


@pytest.mark.remote_data
class TestRegressionOddsRatios:
    """Odds ratios must match baseline within 0.01 log10 units."""

    def test_r_tcb_2Dnmodel_gaiacalctmass(self, baseline, survey_results):
        scalars, _ = baseline
        row = survey_results.fieldstar_models["HIP82545"].candidates
        val = row[row["r_tcb_catalogue"] == "gaiacalctmass"]["r_tcb_2Dnmodel"].values[0]
        assert abs(val - scalars["r_tcb_2Dnmodel_gaiacalctmass"]) < 0.01, (
            f"r_tcb_2Dnmodel (gaiacalctmass): {val:.4f} vs baseline {scalars['r_tcb_2Dnmodel_gaiacalctmass']:.4f}"
        )

    def test_r_tcb_pmmodel_gaiacalctmass(self, baseline, survey_results):
        scalars, _ = baseline
        row = survey_results.fieldstar_models["HIP82545"].candidates
        val = row[row["r_tcb_catalogue"] == "gaiacalctmass"]["r_tcb_pmmodel"].values[0]
        assert abs(val - scalars["r_tcb_pmmodel_gaiacalctmass"]) < 0.01, (
            f"r_tcb_pmmodel (gaiacalctmass): {val:.4f} vs baseline {scalars['r_tcb_pmmodel_gaiacalctmass']:.4f}"
        )

    def test_r_tcb_2Dnmodel_tmass(self, baseline, survey_results):
        scalars, _ = baseline
        row = survey_results.fieldstar_models["HIP82545"].candidates
        val = row[row["r_tcb_catalogue"] == "tmass"]["r_tcb_2Dnmodel"].values[0]
        assert abs(val - scalars["r_tcb_2Dnmodel_tmass"]) < 0.01, (
            f"r_tcb_2Dnmodel (tmass): {val:.4f} vs baseline {scalars['r_tcb_2Dnmodel_tmass']:.4f}"
        )

    def test_r_tcb_pmmodel_tmass(self, baseline, survey_results):
        scalars, _ = baseline
        row = survey_results.fieldstar_models["HIP82545"].candidates
        val = row[row["r_tcb_catalogue"] == "tmass"]["r_tcb_pmmodel"].values[0]
        assert abs(val - scalars["r_tcb_pmmodel_tmass"]) < 0.01, (
            f"r_tcb_pmmodel (tmass): {val:.4f} vs baseline {scalars['r_tcb_pmmodel_tmass']:.4f}"
        )


@pytest.mark.remote_data
class TestRegressionArrays:
    """Means and covariance matrices must match baseline within numerical tolerance."""

    def _get_row(self, survey_results):
        candidates = survey_results.fieldstar_models["HIP82545"].candidates
        return candidates[candidates["r_tcb_catalogue"] == "gaiacalctmass"].iloc[0]

    def test_mean_measured_positions(self, baseline, survey_results):
        _, arrays = baseline
        row = self._get_row(survey_results)
        np.testing.assert_allclose(
            np.array(row["mean_measured_positions"]),
            arrays["mean_measured_positions"],
            rtol=1e-6, atol=1e-6,
            err_msg="mean_measured_positions changed",
        )

    def test_mean_true_companion(self, baseline, survey_results):
        _, arrays = baseline
        row = self._get_row(survey_results)
        np.testing.assert_allclose(
            np.array(row["mean_true_companion"]),
            arrays["mean_true_companion"],
            rtol=1e-6, atol=1e-6,
            err_msg="mean_true_companion changed",
        )

    def test_mean_background_object(self, baseline, survey_results):
        _, arrays = baseline
        row = self._get_row(survey_results)
        np.testing.assert_allclose(
            np.array(row["mean_background_object"]),
            arrays["mean_background_object"],
            rtol=1e-4, atol=1e-4,
            err_msg="mean_background_object changed",
        )

    def test_cov_background_object(self, baseline, survey_results):
        _, arrays = baseline
        row = self._get_row(survey_results)
        np.testing.assert_allclose(
            np.array(row["cov_background_object"]),
            arrays["cov_background_object"],
            rtol=1e-4, atol=1e-4,
            err_msg="cov_background_object changed",
        )

    def test_cov_true_companion(self, baseline, survey_results):
        _, arrays = baseline
        row = self._get_row(survey_results)
        np.testing.assert_allclose(
            np.array(row["cov_true_companion"]),
            arrays["cov_true_companion"],
            rtol=1e-6, atol=1e-6,
            err_msg="cov_true_companion changed",
        )


@pytest.mark.remote_data
class TestRegressionHostStar:
    """Host star Gaia astrometry must be stable (no DR change)."""

    def test_host_pmra(self, baseline, survey_results):
        scalars, _ = baseline
        hs = survey_results.fieldstar_models["HIP82545"]
        assert abs(hs.pmra - scalars["host_pmra"]) < 0.001, (
            f"host pmra: {hs.pmra:.4f} vs baseline {scalars['host_pmra']:.4f}"
        )

    def test_host_pmdec(self, baseline, survey_results):
        scalars, _ = baseline
        hs = survey_results.fieldstar_models["HIP82545"]
        assert abs(hs.pmdec - scalars["host_pmdec"]) < 0.001, (
            f"host pmdec: {hs.pmdec:.4f} vs baseline {scalars['host_pmdec']:.4f}"
        )

    def test_host_parallax(self, baseline, survey_results):
        scalars, _ = baseline
        hs = survey_results.fieldstar_models["HIP82545"]
        assert abs(hs.parallax - scalars["host_parallax"]) < 0.001, (
            f"host parallax: {hs.parallax:.4f} vs baseline {scalars['host_parallax']:.4f}"
        )
