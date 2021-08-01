## This is based on Lehnertz 2018, now implemented by Leonardo Rydin Gorjão and
# Pedro G. Lind for direct retrieval of the jump amplitude and jump rate of a
# jump diffusion process.

import numpy as np

def jump_amplitude(moments: np.ndarray, tol: float = 1e-10,
        full: bool = False, verbose: bool = False) -> np.ndarray:
    r"""
    Retrieves the jump amplitude xi (:math:`\xi`) via

    .. math::

        \lambda(x,t) = \frac{M_4(x,t)}{3\sigma_{\xi}^4}.

    Take notice that the different normalisation of the ``moments`` leads to a
    different results.

    Parameters
    ----------
    moments: np.ndarray
        Moments extracted with the function ``moments``. Needs moments up to
        order ``6``.

    tol: float (defaul ``1e-10``)
        Toleration for the division of the moments.

    full: bool (defaul ``False``)
        If ``True`` returns also the (biased) weighed standard deviation of the
        averaging process.

    verbose: bool (defaul ``True``)
        Prints the result.

    Returns
    -------
    xi_est: np.ndarray
        Estimator of the jump amplitude xi (:math:`\xi`).

    References
    ----------
    Anvari, M., Tabar, M. R. R., Peinke, J., Lehnertz, K., 'Disentangling the
    stochastic behavior of complex time series.' Scientific Reports, 6, 35435,
    2016. doi: 10.1038/srep35435.

    Lehnertz, K., Zabawa, L., and Tabar, M. R. R., 'Characterizing abrupt
    transitions in stochastic dynamics.' New Journal of Physics, 20(11):113043,
    2018. doi: 10.1088/1367-2630/aaf0d7.
    """

    # pre-allocate variable
    xi_est = np.zeros(moments.shape[2])
    xi_est_std = np.zeros(moments.shape[2])


    for i in range(moments.shape[2]):
        mask = moments[0,:,i] < tol

        temp = (moments[6,~mask,i]) / (5 * moments[4,~mask,i])

        xi_est[i] = np.average(temp, weights = moments[0,~mask,i])

        xi_est_std[i] = np.average((temp-xi_est[i])**2,
                            weights = moments[0,~mask,i])

    if verbose == True:
        print(r'ξ = {:f}'.format(xi_est[i]) + r' ± {:f}'.format(xi_est_std[i]))

    if full == True:
        return xi_est, xi_est_std

    if full == False:
        return xi_est


def jump_rate(moments: np.ndarray, xi_est: np.ndarray = None,
              tol: float = 1e-10, full: bool = False,
              verbose: bool = False) -> np.ndarray:
    r"""
    Retrieves the jump rate lamb (:math:`\lambda`) via

    .. math::

        \sigma_{\xi}^2 = \frac{M_6(x,t)}{5M_4(x,t)}.

    Take notice that the different normalisation of the ``moments`` leads to a
    different results.

    Parameters
    ----------
    moments: np.ndarray
        moments extracted with the function 'moments'. Needs moments of order 6.

    tol: float (defaul ``1e-10``)
        Toleration for the division of the moments.

    full: bool (defaul ``False``)
        If ``True`` returns also the (biased) weighed standard deviation of the
        averaging process.

    verbose: bool (defaul ``True``)
        Prints the result.

    Returns
    -------
    xi_est: np.ndarray
        Estimator on the jump rate lamb (:math:`\lambda`)

    References
    ----------
    Anvari, M., Tabar, M. R. R., Peinke, J., Lehnertz, K., 'Disentangling the
    stochastic behavior of complex time series.' Scientific Reports, 6, 35435,
    2016. doi: 10.1038/srep35435.

    Lehnertz, K., Zabawa, L., and Tabar, M. R. R., 'Characterizing abrupt
    transitions in stochastic dynamics.' New Journal of Physics, 20(11):113043,
    2018. doi: 10.1088/1367-2630/aaf0d7.
    """

    # pre-allocate variable
    lamb_est = np.zeros(moments.shape[2])
    lamb_est_std = np.zeros(moments.shape[2])

    # requires knowing the jump amplitude of the process
    if xi_est == None:
        xi_est = jump_amplitude(moments = moments, tol = tol,
                full = False, verbose = False)
    else:
        # is xi_est is not iterable, turn it into a 1-entry array
        xi_est = np.array([xi_est])

    for i in range(moments.shape[2]):
        mask = moments[0,:,i] < tol

        temp = (moments[4,~mask,i]) / (3 * (xi_est[i]**2) )

        lamb_est[i] = np.average(temp, weights = moments[0,~mask,i])

        lamb_est_std[i] = np.average((temp-lamb_est[i])**2,
                            weights=moments[0,~mask,i])

    if verbose == True:
        print((r'λ = {:f}'.format(lamb_est[i]) +
               r' ± {:f}'.format(lamb_est_std[i])))

    if full == True:
        return lamb_est, lamb_est_std

    if full == False:
        return lamb_est
