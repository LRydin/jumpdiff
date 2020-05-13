import numpy as np
from JumpDiff import jdprocess

def test_jdprocess():
    for delta in [1,0.1,0.01,0.001,0.0001]:
        for scheme in ['Euler', 'Milstein']:
            t_final = 1000
            delta_t = delta

            # let us define a drift function
            def a(x):
                return -0.5*x

            # and a (constant) diffusion term
            def b(x):
                return 0.75

            if scheme == 'Milstein':
                def b_prime(x):
                    return 0.

            # Now define a jump amplitude and rate
            xi = 1.5
            lamb = 1.25
            if scheme == 'Euler':
                X = jdprocess(t_final, delta_t, a=a, b=b, xi=xi, lamb=lamb)

            if scheme == 'Milstein':
                X = jdprocess(t_final, delta_t, a=a, b=b, xi=xi, lamb=lamb,
                        b_prime=b_prime, init= 0., solver = scheme)

            assert isinstance(X, np.ndarray)
            assert X.shape[0] == int(t_final/delta_t)
