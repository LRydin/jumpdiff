## This is based on Lehnertz 2018, now implemented by Leonardo Rydin Gorjão and
# Pedro G. Lind for direct application.

import numpy as np
from kramersmoyal import km, kernels

def Qratio(timeseries: np.ndarray, lag: np.ndarray, loc: float=None) -> np.ndarray:
    """
    Qratio method to distinguish pure diffusion from jump-diffusion timeseries,
    introduced by K. Lehnertz, L. Zabawa, and M. Reza Rahimi Tabar.
    'Characterizing abrupt transitions in stochastic dynamics'. New Journal of
    Physics, 20(11):113043, 2018, doi: 10.1088/1367-2630/aaf0d7.

    Parameters
    ----------
    timeseries: np.ndarray
        A 1-dimensional timeseries (N, 1). The timeseries of length N.

    lag: np.ndarray of ints
        An array with the time-lag to extract the Kramers–Moyal coefficient for
        different lags.

    loc: float
        Use a particular point in space to calculate the ratio. If none given,
        the maximum of the probability density function is taken.

    Returns
    -------
    lag: np.ndarray of ints
        Same as input, but only lag > 0 and as ints.

    ratio: np.ndarray of len(lag)
        Ratio of the sixth-order over forth-order Kramers–Moyal coefficient.
    """

    # Force lag to be ints, ensure lag > order + 1
    lag = lag[lag > 0]
    lag = np.round(lag).astype(int)

    # Assert if timeseries is 1 dimensional
    if timeseries.ndim > 1:
        assert timeseries.shape[1] == 1, "Timeseries needs to be 1 dimensional"

    # Initialise function
    ratio = np.zeros(lag.size)

    # Find maxixum of distribution
    if loc == None:
        temp, _ = km(timeseries, powers=np.array([[0]]), bins=np.array([5000]))
        loc = np.argmax(temp[0])

    for i in range(lag.size):
        temp, _ = km(timeseries[::lag[i]], powers=np.array([[4],[6]]),
                     bins=np.array([5000]))
        ratio[i] = temp[1][loc] / (5 * temp[0][loc])

    return lag, ratio
