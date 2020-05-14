## This is based on Lehnertz 2018, now implemented by Leonardo Rydin Gorjão and
# Pedro G. Lind for direct retrieval of the jump amplitude and jump rate of a
# jump diffusion process.

import numpy as np

def jump_amplitude(moments: np.ndarray, tol: float=1e-10,
        full: bool=False, verbose: bool=True) -> np.ndarray:
    """
    Retrieves the jump amplitude xi (ξ) via

                        M₆(x)
          σ²[ξ](x) =  ─────────
                       5 M₄(x)

    Take notice that the different normalisation of the 'moments' leads to a
    different results.

    Parameters
    ----------
    moments: np.ndarray
        moments extracted with the function 'moments'. Needs moments of order 6.

    tol: float
        Toleration for the division of the moments.

    full: bool
        If 'True' returns also the (biased) weighed standard deviation of the
        averaging process.

    Returns
    -------
    xi_est: np.ndarray
        Estimator of the jump amplitude xi (ξ).
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


def jump_rate(moments: np.ndarray, xi_est: np.ndarray=None, tol: float=1e-10,
        full: bool=False, verbose: bool=True) -> np.ndarray:
    """
    Retrieves the jump rate lamb (λ) via

                      M₄(x)
          λ(x) =  ────────────
                   3 σ²[ξ](x)

    which requires σ²[ξ](x), i.e., find 'jump_amplitude()' before.

    Parameters
    ----------
    moments: np.ndarray
        moments extracted with the function 'moments'. Needs moments of order 6.

    xi_est: np.ndarray
        If the amplitude of the jumps are known, they can be added manually,
        else by default it will first retrieve the jump amplitude and then
        calculates the jump rate.

    tol: float
        Toleration for the division of the moments.

    full: bool
        If 'True' returns also the (biased) weighed standard deviation of the
        averaging process.

    Returns
    -------
    lamb_est: float
        Estimator on the jump rate lamb (λ)
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
        print(r'λ = {:f}'.format(lamb_est[i]) + r' ± {:f}'.format(lamb_est_std[i]))

    if full == True:
        return lamb_est, lamb_est_std

    if full == False:
        return lamb_est
