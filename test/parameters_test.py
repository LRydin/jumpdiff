import numpy as np
from JumpDiff import jdprocess, jump_amplitude, jump_rate, moments

def test_parameters():
    for delta in [1,0.1,0.01,0.001,0.0001]:
        for full in [True, False]:
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

            edges, m = moments(timeseries = X)

            if full == False:
                xi_est = jump_amplitude(moments = m, full = full)
                lamb_est = jump_rate(moments = m, full = full)

                assert isinstance(lamb_est, np.ndarray)
                assert isinstance(xi_est, np.ndarray)

            if full == True:
                xi_est, xi_est_std = jump_amplitude(moments = m, full = full)
                lamb_est, lamb_est_std = jump_rate(moments = m, full = full)

                assert isinstance(lamb_est, np.ndarray)
                assert isinstance(xi_est, np.ndarray)
                assert isinstance(lamb_est_std, np.ndarray)
                assert isinstance(xi_est_std, np.ndarray)
