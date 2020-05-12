## This was developed originally by Francisco Meirinhos and Leonardo Rydin
# Gorjão for kernel-density estimation of Kramers–Moyal coefficients in
# https://github.com/LRydin/KramersMoyal and published in 'kramersmoyal:
# Kramers--Moyal coefficients for stochastic processes'. Journal of Open Source
# Software, 4(44), 1693, doi: 10.21105/joss.01693. It is now extend by Leonardo
# Rydin Gorjão and Pedro G. Lind to include second-order corrections of the
# Kramers─Moyal conditional moments, as well as a built-in lag to calculate the
# moments at different timesteps.

import numpy as np
from scipy.signal import convolve
from scipy.special import factorial

from .binning import histogramdd
from .kernels import silvermans_rule, epanechnikov, _kernels

def moments(timeseries: np.ndarray, bins: np.ndarray=None, power: int=6,
        lag: list=[1], correction: bool=True, norm: bool=False,
        kernel: callable=None, bw: float=None, tol: float=1e-10,
        conv_method: str='auto'):
    """
    Estimates the moments of the Kramers─Moyal expansion from a timeseries using
    a Nadaraya─Watson kernel estimator method. These later can be turned into
    the drift and diffusion coefficients after normalisation.

    Parameters
    ----------
    timeseries: np.ndarray
        A 1-dimensional timeseries of length N.

    bins: np.ndarray
        The number of bins for each dimension, defaults to 'np.array([5000])'.
        This is the underlying space for the Kramers─Moyal conditional moments.

    power: int
        Upper limit of the the Kramers─Moyal conditional moments to calculate.
        It will generate all Kramers─Moyal conditional moments up to power.

    lag: list
        Calculates the Kramers─Moyal conditional moments at each indicated lag,
        i. e., for timeseries[::lag[]]. Defaults to '1', the shortest timestep
        in the data.

    corrections: bool
        Implements the second-order corrections of the Kramers─Moyal conditional
        moments directly

    norm: bool
        Sets the normalisation. 'False' returns the Kramers─Moyal conditional
        moments, and 'True' returns the Kramers─Moyal coefficients.

    kernel: callable
        Kernel used to convolute with the Kramers─Moyal conditional moments. To
        select example an Epanechnikov kernel use
            kernel = kernels.epanechnikov
        If None the Epanechnikov kernel will be used.

    bw: float
        Desired bandwidth of the kernel. A value of 1 occupies the full space of
        the bin space. Recommended are values 0.005 < bw < 0.4.

    tol: float
        Round to zero absolute values smaller than `tol`, after convolutions.

    conv_method: str
        A string indicating which method to use to calculate the convolution.
        docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve.

    Returns
    -------
    edges: np.ndarray
        The bin edges with shape (D,bins.shape) of the calculated moments.

    moments: np.ndarray
        The calculated moments from the Kramers─Moyal expansion of the
        timeseries at each lag. To extract the selected orders of the moments,
        use moments[i,:,j], with i the order according to powers, j the lag (if
        any given)
    """
    timeseries = np.asarray_chkfinite(timeseries, dtype=float)
    if len(timeseries.shape) == 1:
        timeseries = timeseries.reshape(-1, 1)

    assert len(timeseries.shape) == 2, "Timeseries must (n, dims) shape"
    assert timeseries.shape[0] > 0, "No data in timeseries"

    n, dims = timeseries.shape

    if bins is None:
        bins = np.array([5000])

    if lag is None:
        lag = [1]

    powers = np.linspace(0,power,power+1).astype(int)
    if len(powers.shape) == 1:
        powers = powers.reshape(-1, 1)

    assert (powers[0] == [0] * dims).all(), "First power must be zero"
    assert dims == powers.shape[1], "Powers not matching timeseries' dimension"
    assert dims == bins.shape[0], "Bins not matching timeseries' dimension"

    if bw is None:
        bw = silvermans_rule(timeseries)*2.
    elif callable(bw):
        bw = bw(timeseries)
    assert bw > 0.0, "Bandwidth must be > 0"

    if kernel is None:
        kernel = epanechnikov
    assert kernel in _kernels, "Kernel not found"

    edges, moments =  _moments(timeseries, bins, powers, lag, kernel, bw, tol, conv_method)

    if correction == True:
        moments = corrections(m = moments, power = power)

    if norm == True:
        for i in range(power):
            moments = moments / float(factorial(i))


    return (edges, moments)


def _moments(timeseries: np.ndarray, bins: np.ndarray, powers: np.ndarray,
        lag: list, kernel: callable, bw: float, tol: float, conv_method: str):
    """
    Helper function for km that does the heavy lifting and actually estimates
    the Kramers─Moyal coefficients from the timeseries.
    """
    def cartesian_product(arrays: np.ndarray):
        # Taken from https://stackoverflow.com/questions/11144513
        la = len(arrays)
        arr = np.empty([len(a) for a in arrays] + [la], dtype=np.float64)
        for i, a in enumerate(np.ix_(*arrays)):
            arr[..., i] = a
        return arr.reshape(-1, la)

    def kernel_edges(edges: np.ndarray):
        # Generates the kernel edges
        edges_k = list()
        for edge in edges:
            dx = edge[1] - edge[0]
            L = edge.size
            edges_k.append(np.linspace(-dx * L, dx * L, int(2 * L + 1)))
        return edges_k

    # Calculate derivative and the product of its powers

    grads = np.diff(timeseries, axis=0)
    weights = np.prod(np.power(grads.T, powers[..., None]), axis=1)

    # Get weighted histogram
    hist, edges = histogramdd(timeseries[:-1, ...], bins=bins,
                              weights=weights, bw=bw)

    # Generate centred kernel
    edges_k = kernel_edges(edges)
    mesh = cartesian_product(edges_k)
    kernel_ = kernel(mesh, bw=bw).reshape(*(edge.size for edge in edges_k))
    kernel_ /= np.sum(kernel_)

    # Convolve weighted histogram with kernel and trim it
    kmc_temp = convolve(hist, kernel_[None, ...], mode='same', method=conv_method)

    moments = np.zeros(kmc_temp.shape + (len(lag),))
    edge_ = np.zeros(edges[0][:-1].shape + (len(lag),))

    for i in range(len(lag)):
        ts = timeseries[::lag[i]]
        grads = np.diff(ts, axis=0)
        weights = np.prod(np.power(grads.T, powers[..., None]), axis=1)

        # Get weighted histogram
        hist, edges = histogramdd(ts[:-1, ...], bins=bins,
                                  weights=weights, bw=bw)

        # Convolve weighted histogram with kernel and trim it
        kmc = convolve(hist, kernel_[None, ...], mode='same', method=conv_method)

        # Normalise
        mask = np.abs(kmc[0]) < tol
        kmc[0:, mask] = 0.0
        kmc[1:, ~mask] /= kmc[0, ~mask]

        # Pack moments and edges here
        moments[..., i] = kmc
        edge_[...,i] = [edge[:-1] + 0.5*(edge[1] - edge[0]) for edge in edges][0]


    return edge_, moments

def corrections(m, power, norm: bool=False):
    """
    The moments function will by default apply the corrections. You can turn
    the corrections off in that fuction by setting 'corrections = False'.

    Second-order corrections of the Kramers─Moyal coefficients (conditional
    moments), given by

        F₁ =       M₁
        F₂ =   1/2(M₂ - M₁²)
        F₃ =   1/6(M₃ - 3M₁M₂ + 3M₁³)
        F₄ =  1/24(M₄ - 4M₁M₃ + 18M₁²M₂ - 3M₂² - 15M₁⁴)
        F₅ = 1/120(M₅ - 5M₁M₄ + 30M₁²M₃ - 150M₁³M₂ + 45M₁M₂² - 10M₂M₃ + 105M₁⁵)
        F₆ = 1/720(M₆ - 6M₁M₅ + 45M₁²M₄ - 300M₁³M₃ + 1575M₁⁴M₂ - 675M₁²M₂²
                    + 180M₁M₂M₃ + 45M₂³ - 15M₂M₄ - 10M₃² - 945M₁⁶)

    with the prefactor the normalisation, i.e., the normalised results are the
    Kramers─Moyal coefficients. If 'norm' is False, this results in the
    Kramers─Moyal conditional moments.

    Parameters
    ----------
    m (moments): np.ndarray
        The calculated conditional moments from the Kramers─Moyal expansion of
        the at each lag. To extract the selected orders of the moments use
        moments[i,:,j], with i the order according to powers, j the lag.

    power: int
        Upper limit of the Kramers─Moyal conditional moments to calculate.
        It will generate all Kramers─Moyal conditional moments up to power.

    Returns
    -------
    F: np.ndarray
        The corrections of the calculated Kramers─Moyal conditional moments
        from the Kramers─Moyal expansion of the timeseries at each lag. To
        extract the selected orders of the moments, use F[i,:,j], with i the
        order according to powers, j the lag (if any introduced).
    """

    powers = np.linspace(0,power,power+1).astype(int)
    if len(powers.shape) == 1:
        powers = powers.reshape(-1, 1)

    # remnant of the normalisation factor, moved to 'moments'
    normalise = np.ones_like(powers)

    F = np.zeros_like(m)

    if 0 in powers:
        F[0] = normalise[0] * m[0]
    if 1 in powers:
        F[1] = normalise[1] * m[1]
    if 2 in powers:
        F[2] = normalise[2] * ( m[2] - (m[1]**2) )
    if 3 in powers:
        F[3] = normalise[3] * ( m[3] - 3*m[1]*m[2] + 3*(m[1]**3) )
    if 4 in powers:
        F[4] = normalise[4] * ( m[4] - 4*m[1]*m[3] + 18*(m[1]**2)*m[2] \
            - 3*m[2]**2 - 15*(m[1]**4) )
    if 5 in powers:
        F[5] = normalise[5] * ( m[5] - 5*m[1]*m[4] + 30*(m[1]**2)*m[3] \
            - 150*(m[1]**3)*m[2] + 45*m[1]*(m[2]**2) - 10*m[2]*m[3] \
            + 105*(m[1]**5) )
    if 6 in powers:
        F[6] = normalise[6] * ( m[6] - 6*m[1]*m[5] + 45*(m[1]**2)*m[4] \
            - 300*(m[1]**3)*m[3] + 1575*(m[1]**4)*m[2] \
            - 675*(m[1]**2)*(m[2]**2) + 180*m[1]*m[2]*m[3] \
            + 45*(m[3]**3) - 15*m[2]*m[4] - 10*(m[3]**2) \
            - 945*(m[1]**6) )

    return F
