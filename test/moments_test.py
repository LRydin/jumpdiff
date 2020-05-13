import numpy as np
from JumpDiff import jdprocess, moments

def test_moments():
    for delta in [1,0.1,0.01,0.001]:
        for lag in [None, [1,2,3]]:
            t_final = 1000
            delta_t = delta

            # let us define a drift function
            def a(x):
                return -0.5*x

            # and a (constant) diffusion term
            def b(x):
                return 0.75


            # Now define a jump amplitude and rate
            xi = 1.5
            lamb = 1.25

            X = jdprocess(t_final, delta_t, a = a, b = b, xi = xi, lamb = lamb)

            edges, m = moments(timeseries = X, lag = lag)

            assert isinstance(edges, np.ndarray)
            assert isinstance(m, np.ndarray)

            edges, m = moments(timeseries = X, lag = lag, bw = 0.3)

            assert isinstance(edges, np.ndarray)
            assert isinstance(m, np.ndarray)

            edges, m = moments(timeseries = X, lag = lag, norm = True)

            assert isinstance(edges, np.ndarray)
            assert isinstance(m, np.ndarray)
