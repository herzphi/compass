import numpy as np
import pytest

from compass.helperfunctions import get_ellipse_props, parallax_projection
from compass.model import HostStar, CovarianceMatrix


def test_get_ellipse_props():
    """Test example."""
    assert get_ellipse_props(np.array([[1, 0], [0, 1]]), 0.8) == (
        1.7941225779941015,
        1.7941225779941015,
        0.0,
    )


def test_host_star_object():
    """Test the host star object wether all parameters are given."""
    host_star = HostStar(target="HIP82545")
    attribute_list = [
        "object_found",
        "ra",
        "ra_error",
        "dec",
        "dec_error",
        "ref_epoch",
        "parallax",
        "parallax_error",
        "pmra",
        "pmdec",
        "pmra_error",
        "pmdec_error",
        "pmra_pmdec_corr",
        "parallax_pmra_corr",
        "parallax_pmdec_corr",
        "phot_g_mean_mag",
        "phot_bp_mean_mag",
        "phot_rp_mean_mag",
    ]
    assert list(host_star.__dict__) == attribute_list
    host_star.cone_gaia_objects(0.1)
    assert len(host_star.cone_gaia) == 9257
    df_bp = host_star.concat_binning_parameters(host_star.cone_gaia, "ks_m_calc")
    host_star.calc_background_model_parameters([df_bp], "band", None, False)
    assert len(list(host_star.__dict__)) == 49


def test_covariancematrix():
    host_star = HostStar(target="HIP82545")
    times_in_days = [800, 1900]
    time_days = np.linspace(0, times_in_days[1] + 1, int(times_in_days[1] + 1))
    plx_proj_ra, plx_proj_dec = parallax_projection(time_days, host_star)
    cov = CovarianceMatrix.covariance_matrix(
        times_in_days, plx_proj_ra, plx_proj_dec, host_star
    ).round(2)
    assert (
        cov
        == np.array(
            [
                [0.46, -0.03, 1.02, -0.08],
                [-0.03, 0.32, -0.08, 0.81],
                [1.02, -0.08, 2.43, -0.19],
                [-0.08, 0.81, -0.19, 1.93],
            ]
        )
    ).all()
