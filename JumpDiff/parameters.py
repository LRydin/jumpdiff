## This is based on Lehnertz 2018, now implemented by Leonardo Rydin Gorjão and
# Pedro G. Lind for direct retrieval of the jump amplitude and jump rate of a
# jump diffusion process.

import numpy as np

def jump_amplitude(moments: np.ndarray, tol: float=1e-10,
        full: bool=False) -> np.ndarray:
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

    xi_est = np.zeros(moments.shape[2])

    if full == True:
        xi_est_std = np.zeros(moments.shape[2])


    for i in range(moments.shape[2]):
        mask = moments[0,:,i] < tol

        temp = (moments[6,~mask,i]) / (5 * moments[4,~mask,i])

        xi_est[i] = np.average(temp, weights = np.sqrt(moments[0,~mask,i]))
        # xi_est[i] = np.average(temp)

        if full == True:
            xi_est_std[i] = np.average((temp-xi_est[i])**2,
                                weights = moments[0,~mask,i])


    if full == True:
        return xi_est, xi_est_std

    if full == False:
        return xi_est


def jump_rate(moments: np.ndarray, xi_est: np.array, tol: float=1e-10,
        full: bool=False) -> np.ndarray:
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

    # is xi_est is not iterable, turn it into a 1-entry array
    xi_est = np.array([xi_est])

    lamb_est = np.zeros(moments.shape[2])

    if full == True:
        lamb_est_std = np.zeros(moments.shape[2])

    for i in range(moments.shape[2]):
        mask = moments[0,:,i] < tol

        temp = (moments[4,~mask,i]) / (3 * (xi_est[i]**2) )

        lamb_est[i] = np.average(temp,  weights = np.sqrt(moments[0,~mask,i]))
        # lamb_est[i] = np.average(temp)

        if full == True:
            lamb_est_std[i] = np.average((temp-lamb_est[i])**2,
                                weights=moments[0,~mask,i])

    if full == True:
        return lamb_est, lamb_est_std

    if full == False:
        return lamb_est
