import numpy as np
from scipy.stats import binom
from scipy.integrate import quad


def calculate_my_likelihood(d_g, n_g, p_g, prob_dens_func, w_g, gamma_g):
    """
    Numerically calculates the value of L(d_g) based on the given formula.

    Parameters:
        d_g (int): Value of d_g.
        n_g (int): Value of n_g.
        p_g (callable): The p_g function representing the probability density function.
        prob_dens_func (callable): The pdf_g function representing the probability density function.
        w_g (float): Parameter 'w_g'.
        gamma_g (float): Parameter 'gamma_g'.

    Returns:
        float: Numerical approximation of the integral.
    """

    integrand = lambda x: binom.pmf(d_g, n_g, p_g(x, w_g, gamma_g)) * prob_dens_func(x)

    result, _ = quad(integrand, -5, 5)

    return result


def calculate_likelihood_ts(d_g, n_g, p_g, pdf_g, w_g, gamma_g):
    """
    Numerically calculates the time series value of L(d_g) based on the given formula by multiply for each date.

    Parameters:
        d_g (pd.Series): Time series for d_g.
        n_g (pd.Series): Time series for n_g.
        p_g (callable): The p_g function representing the probability density function.
        pdf_g (callable): The pdf_g function representing the probability density function.
        w_g (float): Parameter 'w_g'.
        gamma_g (float): Parameter 'gamma_g'.

    Returns:
        float: Numerical approximation of the integral.
    """
    integrand = lambda x: np.prod(binom.pmf(d_g, n_g, p_g(x, w_g, gamma_g)) * pdf_g(x))

    result, _ = quad(integrand, -5, 5)

    return result
