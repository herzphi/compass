import pytest
import numpy as np
from compass import modelling


def test_get_ellipse_props():
    """Test example.
    """
    assert modelling.HelperFunctions.get_ellipse_props(
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
