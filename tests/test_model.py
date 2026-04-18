import numpy as np
import pytest

from compass import helperfunctions
from compass import model


def test_get_ellipse_props():
    """Test example."""
    assert helperfunctions.get_ellipse_props(np.array([[1, 0], [0, 1]]), 0.8) == (
        1.7941225779941015,
        1.7941225779941015,
        0.0,
    )


@pytest.mark.remote_data
def test_host_star_object():
    """Test the host star object wether all parameters are given."""
    host_star = model.HostStar(target="HIP82545")
    # Verify all expected Gaia attributes are present after construction
    expected_attrs = {
        "object_found",
        "background_model_coeffs",
        "binning_parameters_tables",
        "ra", "ra_error", "dec", "dec_error", "ref_epoch",
        "parallax", "parallax_error",
        "pmra", "pmdec", "pmra_error", "pmdec_error",
        "pmra_pmdec_corr", "parallax_pmra_corr", "parallax_pmdec_corr",
        "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag",
    }
    assert expected_attrs.issubset(set(host_star.__dict__))
    assert list(host_star.__dict__)[0] == "object_found"
    host_star.cone_gaia_objects(0.1)
    assert len(host_star.cone_gaia) == 9257
    df_bp = host_star.concat_binning_parameters(
        host_star.cone_gaia, "ks_m_calc", binsize=100
    )
    host_star.calc_background_model_parameters([df_bp], "band", None, False)
    # Background model coefficients are now stored in background_model_coeffs dict
    assert "gaiacalc" in host_star.background_model_coeffs
    assert "pmra_mean_coeff" in host_star.background_model_coeffs["gaiacalc"]
