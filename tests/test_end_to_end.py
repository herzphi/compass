"""End-to-end integration test using the example from the README.

Uses synthetic 3-epoch astrometry of a candidate near HIP82545.
The candidate has near-zero relative proper motion (dRA changes < 10 mas
over 5 years) so it is expected to favour the companion model.
Requires network access to Gaia and Simbad.
"""
import numpy as np
import pandas as pd
import pytest

from compass import model


# Synthetic data from README: 3-epoch candidate near HIP82545.
# Separation ~1700 mas, nearly static relative position → expected companion.
README_DATA = pd.DataFrame(
    {
        "Main_ID": ["HIP82545", "HIP82545", "HIP82545"],
        "final_uuid": ["id0", "id0", "id0"],
        "date": ["2016-04-24", "2018-04-24", "2021-06-04"],
        "dRA": [-1390, -1386, -1394],
        "dRA_err": [5, 5, 5],
        "dDEC": [-990, -989, -995],
        "dDEC_err": [5, 5, 5],
        "dRA_dDEC_corr": [0, 0, 0],
        "mag0": [16, 16, 16],
        "mag0_err": [1, 1, 1],
    }
)


@pytest.mark.remote_data
class TestSurveyEndToEnd:
    """Full pipeline: Survey init → field star model → evaluation → results."""

    @pytest.fixture(scope="class")
    def survey(self):
        """Run the full pipeline once and share the result across tests."""
        sv = model.Survey(README_DATA, "mag0")
        sv.set_fieldstar_models("ks_m_calc", "ks_m", cone_radius=0.1, binsize=50)
        sv.set_evaluated_fieldstar_models(sigma_cc_min=0, sigma_model_min=0)
        return sv

    # ------------------------------------------------------------------
    # Survey initialisation
    # ------------------------------------------------------------------

    def test_target_registered(self, survey):
        assert "HIP82545" in survey.target_names

    def test_candidates_data_created(self, survey):
        df = survey.candidates_data_HIP82545
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1  # one unique final_uuid
        assert "id0" in df["final_uuid"].values

    def test_candidate_relative_pm_computed(self, survey):
        df = survey.candidates_data_HIP82545
        # pmra_mean = (dRA_last - dRA_first) / dt  ≈ small number
        assert np.isfinite(df["pmra_mean"].values[0])
        assert np.isfinite(df["pmdec_mean"].values[0])

    # ------------------------------------------------------------------
    # Field star model
    # ------------------------------------------------------------------

    def test_fieldstar_model_created(self, survey):
        assert hasattr(survey, "fieldstar_model_HIP82545")
        host_star = survey.fieldstar_model_HIP82545
        # Check fitted model coefficients exist for the combined catalogue
        assert hasattr(host_star, "pmra_mean_model_coeff_gaiacalctmass")
        assert hasattr(host_star, "pmdec_mean_model_coeff_gaiacalctmass")
        assert hasattr(host_star, "pmra_stddev_model_coeff_gaiacalctmass")
        assert hasattr(host_star, "parallax_mean_model_coeff_gaiacalctmass")

    def test_fieldstar_model_coefficients_finite(self, survey):
        host_star = survey.fieldstar_model_HIP82545
        for attr in [
            "pmra_mean_model_coeff_gaiacalctmass",
            "pmdec_mean_model_coeff_gaiacalctmass",
            "pmra_stddev_model_coeff_gaiacalctmass",
            "pmdec_stddev_model_coeff_gaiacalctmass",
        ]:
            coeffs = getattr(host_star, attr)
            assert np.all(np.isfinite(coeffs)), f"{attr} contains non-finite values"

    # ------------------------------------------------------------------
    # Candidate evaluation
    # ------------------------------------------------------------------

    def test_candidates_evaluated(self, survey):
        host_star = survey.fieldstar_model_HIP82545
        assert hasattr(host_star, "candidates")
        assert isinstance(host_star.candidates, pd.DataFrame)
        assert len(host_star.candidates) > 0

    def test_odds_ratio_2d_finite(self, survey):
        host_star = survey.fieldstar_model_HIP82545
        r = host_star.candidates["r_tcb_2Dnmodel"]
        assert r.notna().all(), "r_tcb_2Dnmodel contains NaN"
        assert np.isfinite(r).all(), "r_tcb_2Dnmodel contains inf"

    def test_odds_ratio_pm_finite(self, survey):
        host_star = survey.fieldstar_model_HIP82545
        r = host_star.candidates["r_tcb_pmmodel"]
        assert r.notna().all(), "r_tcb_pmmodel contains NaN"
        assert np.isfinite(r).all(), "r_tcb_pmmodel contains inf"

    def test_nearly_static_candidate_favours_companion(self, survey):
        """Candidate with <10 mas drift over 5 yr should have positive odds ratio."""
        host_star = survey.fieldstar_model_HIP82545
        gaiacalctmass = host_star.candidates[
            host_star.candidates["r_tcb_catalogue"] == "gaiacalctmass"
        ]
        r = gaiacalctmass["r_tcb_2Dnmodel"].values[0]
        assert r > 0, (
            f"Expected positive odds ratio for near-static candidate, got {r:.3f}"
        )

    def test_covariance_matrices_correct_shape(self, survey):
        host_star = survey.fieldstar_model_HIP82545
        # 3 epochs → 2N = 6
        row = host_star.candidates.iloc[0]
        assert np.array(row["cov_background_object"]).shape == (6, 6)
        assert np.array(row["cov_true_companion"]).shape == (6, 6)
        assert np.array(row["cov_measured_positions"]).shape == (6, 6)

    def test_get_true_companions_returns_dataframe(self, survey):
        result = survey.get_true_companions(threshold=0)
        assert isinstance(result, pd.DataFrame)
        assert "r_tcb_2Dnmodel" in result.columns
        assert "final_uuid" in result.columns
        # All returned rows must be above threshold
        assert (result["r_tcb_2Dnmodel"] > 0).all()
