import pytest
import numpy as np
from compass.modelling import HelperFunctions

def test_type():
    """
    Test addition with float, int and strung.
    """
    assert HelperFunctions.get_ellipse_props(np.array([[1,0],[0,1]]), .8) == (1.7941225779941015, 1.7941225779941015, 0.0)

