import numpy as np
from JumpDiff import Qratio, jdprocess

def test_Qratio():
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

        lag = np.unique(np.logspace(0, np.log10(int(t_final/delta_t) // 100), 200).astype(int)+1)

        _, ratio = Qratio(lag = lag, timeseries = X)

        assert isinstance(ratio, np.ndarray)
        assert ratio.shape[0] == lag.shape[0]
