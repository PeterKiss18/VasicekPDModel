import numpy as np
from scipy.stats import norm


def p_g(x_g, w_g, gamma_g):
    """
    Calculates the success probability of a Bernoulli trial.

    Parameters:
        x_g (float): Input value.
        w_g (float): Parameter 'w_g'.
        gamma_g (float): Parameter 'gamma_g'.

    Returns:
        float: Result of p_g(X_g).
    """
    result = norm.cdf((gamma_g - w_g * x_g) / np.sqrt(1 - w_g**2))

    return result
