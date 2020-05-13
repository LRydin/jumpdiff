## This is based on Lehnertz 2018, now implemented by Leonardo Rydin Gorjão and
# Pedro G. Lind for direct application.

import numpy as np
from .moments import moments

def Qratio(lag: np.ndarray, timeseries: np.ndarray, loc: int=None,
        correction: bool=True) -> np.ndarray:
    """
    Qratio method to distinguish pure diffusion from jump-diffusion timeseries,
    Given by the relation of the 4th and 6th Kramers─Moyal coefficient with
    increasing lag

                      D₆(x,τ)
          Q(x,τ) =  ─────────── ,
                     5 D₄(x,τ)

    introduced by K. Lehnertz, L. Zabawa, and M. Reza Rahimi Tabar in
    'Characterizing abrupt transitions in stochastic dynamics'. New Journal of
    Physics, 20(11):113043, 2018, doi: 10.1088/1367-2630/aaf0d7.

    Parameters
    ----------
    lag: np.ndarray of ints
        An array with the time-lag to extract the Kramers–Moyal coefficient for
        different lags.

    timeseries: np.ndarray
        A 1-dimensional timeseries (N, 1). The timeseries of length N.

    loc: float
        Use a particular point in space to calculate the ratio. If none given,
        the maximum of the probability density function is taken.

    corrections: bool
        Implements the second-order corrections of the Kramers─Moyal conditional
        moments directly. Default 'False', since the Q-ratio is only proven at
        first-order.

    Returns
    -------
    lag: np.ndarray of ints
        Same as input, but only lag > 0 and as ints.

    ratio: np.ndarray of len(lag)
        Ratio of the sixth-order over forth-order Kramers–Moyal coefficient.
    """

    # Force lag to be ints, ensure lag > order + 1, and removes duplicates
    lag = lag[lag > 0]
    lag = np.round(np.unique(lag)).astype(int)

    # Assert if timeseries is 1 dimensional
    if timeseries.ndim > 1:
        assert timeseries.shape[1] == 1, "Timeseries needs to be 1 dimensional"

    # Find maximum of distribution
    if loc == None:
        temp = moments(timeseries, power=0, bins=np.array([5000]))[1]
        loc = np.argmax(temp[0])

    temp = moments(timeseries, power=6, bins=np.array([5000]), lag = lag,
        correction = correction)[1]
    ratio = temp[6,loc,:]/(5 * temp[4,loc,:])

    return lag, ratio
