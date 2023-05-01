import nafi

import numpy as np


def test_maximum_likelihood():

    # Some random parabola; maximum is x= -2.5, y = 5.5
    def f(x):
        return - (0.4 * x**2 + 2 * x - 3)
    
    # Evaluate it only on a grid
    x = np.arange(-4, 4)
    y = f(x)

    # Test parabolic interpolation is exact 
    assert np.allclose(nafi.maximum_likelihood(y[None,:])[0], 5.5)
