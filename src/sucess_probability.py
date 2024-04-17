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
    w_g = np.asarray([w_g]) if isinstance(w_g, float) else np.asarray(w_g)
    gamma_g = np.asarray([gamma_g]) if isinstance(gamma_g, float) else np.asarray(gamma_g)

    x_g = np.asarray([x_g]) if isinstance(x_g, float) else np.asarray(x_g)

    x_dim = len(x_g)
    w_dim = len(w_g)

    # Make w_g a x_dim x w_dim matrix where each row is the same
    w_g = np.tile(w_g, (x_dim, 1))
    # Make gamma_g a x_dim x w_dim matrix where each row is the same
    gamma_g = np.tile(gamma_g, (x_dim, 1))
    # Make x_g a x_dim x w_dim matrix where each column is the same
    x_g = np.tile(x_g, (w_dim, 1)).T

    result = norm.cdf((gamma_g - w_g * x_g) / np.sqrt(1 - w_g**2))

    return result
