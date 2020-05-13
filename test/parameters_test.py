import numpy as np
from JumpDiff import jdprocess, jump_amplitude, jump_rate, moments

def test_parameters():
    for delta in [1,0.1,0.01,0.001,0.0001]:
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

        # and simply call the integration function
        X = jdprocess(t_final, delta_t, a=a, b=b, xi=xi, lamb=lamb)

        edges, moments = jd.moments(timeseries = X)

        xi_est = jd.jump_amplitude(moments = moments)

        lamb_est = jd.jump_rate(moments = moments, xi_est = xi)

        if not isinstance(lamb_est, np.ndarray):
            raise Exception('Results is not an array')

        if not isinstance(xi_est, np.ndarray):
            raise Exception('Results is not an array')
