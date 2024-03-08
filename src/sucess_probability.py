import numpy as np
from scipy.stats import norm


def p_g(x_g, w_g, gamma_g):
    """
    Calculates the success probability of a Bernoulli trial.

    Parameters:
        x_g (float): Input value.
        w_g (float or array-like): Parameter 'w_g'.
        gamma_g (float or array-like): Parameter 'gamma_g'.

    Returns:
        float or ndarray: Result of p_g(X_g).
    """
    # Ensure w_g and gamma_g are arrays for element-wise operations
    w_g = np.asarray(w_g)
    gamma_g = np.asarray(gamma_g)

    result = norm.cdf((gamma_g - w_g * x_g) / np.sqrt(1 - w_g**2))

    return result
