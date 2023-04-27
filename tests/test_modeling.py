import pytest
import numpy as np
from compass import modelling
from compass import helperfunctions


def test_get_ellipse_props():
    """Test example.
    """
    assert helperfunctions.get_ellipse_props(
        np.array([[1, 0], [0, 1]]),
        .8
    ) == (1.7941225779941015, 1.7941225779941015, 0.0)


def test_host_star_object():
    """Test the host star object wether all parameters are given."""
    host_star = modelling.HostStar(target='HIP82545')
    attribute_list = ['object_found', 'ra', 'ra_error', 'dec', 'dec_error',
                      'ref_epoch', 'parallax', 'parallax_error', 'pmra',
                      'pmdec', 'pmra_error', 'pmdec_error', 'pmra_pmdec_corr',
                      'parallax_pmra_corr', 'parallax_pmdec_corr',
                      'phot_g_mean_mag', 'phot_bp_mean_mag',
                      'phot_rp_mean_mag']
    assert list(host_star.__dict__) == attribute_list
    host_star.cone_gaia_objects(.1)
    assert len(host_star.cone_gaia) == 9257
    df_bp = host_star.concat_binning_parameters(
        host_star.cone_gaia,
        'ks_m_calc'
    )
    host_star.pmm_parameters(
        [df_bp],
        'band',
        None,
        False
    )
    assert len(list(host_star.__dict__)) == 49


def test_covariancematrix():
    host_star = modelling.HostStar(target='HIP82545')
    times = [800, 1900]
    time_days = np.linspace(
            0,
            times[1]+1,
            int(times[1]+1)
        )
    plx_proj_ra, plx_proj_dec = helperfunctions.parallax_projection(time_days, host_star)
    cov = modelling.CovarianceMatrix.covariance_matrix(
            times,
            plx_proj_ra,
            plx_proj_dec,
            host_star
        )
    assert cov == np.array([
        [0.43223405, -0.03421271,  1.02463317, 0.08125518],
        [-0.03421271,  0.37732801, -0.08125518, 0.80727282],
        [1.02463317, -0.08125518,  2.47563848, 0.19298105],
        [-0.08125518,  0.80727282, -0.19298105, 1.86715406]]
    )