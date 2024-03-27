import numpy as np
import pandas as pd
from scipy.stats import binom, norm
from scipy.integrate import quad
from scipy.optimize import minimize


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

    result, _ = quad(integrand, -3, 3)

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


def calculate_my_likelihood_arr(d_g_arr, n_g_arr, p_g, prob_dens_func, w_g_arr, gamma_g_arr):
    """
    Numerically calculates the value of L(d_g_arr) for multiple grades based on the given formula.

    Parameters:
        d_g_arr (numpy.array(int)): Values of d_g's by grades
        n_g_arr (numpy.array(int)): Values of n_g's by grades
        p_g (callable): The p_g function representing the probability density function.
        prob_dens_func (callable): The pdf_g function representing the probability density function.
        w_g_arr (numpy.array(float)): Parameter 'w_g's by grades
        gamma_g_arr (numpy.array(float)): Parameter 'gamma_g's by grades.

    Returns:
        float: Numerical approximation of the integral.
    """

    integrand = lambda x: np.prod(binom.pmf(d_g_arr, n_g_arr, p_g(x, w_g_arr, gamma_g_arr))) * prob_dens_func(x)

    result, _ = quad(integrand, -3, 3)

    return result


def maximum_likelihood_estimation(d_g, n_g, prob_dens_func, p_g, w_initial, gamma_initial, bounds):
    """
    Estimate w_g and gamma_g using the maximum likelihood estimation method.

    Parameters:
        d_g (pd.Series): Time series for d_g.
        n_g (pd.Series): Time series for n_g.
        prob_dens_func (callable): The pdf_g function representing the probability density function.
        p_g (callable): The p_g function representing the probability density function.
        w_initial (float): Initial guess for w_g.
        gamma_initial (float): Initial guess for gamma_g.
        bounds (list): Bounds for the minimization algorithm.

    Returns:
        tuple: Estimated w_g and gamma_g.
    """

    # Sum of d_g and n_g if they are pd.Series
    if isinstance(d_g, pd.Series):
        d_total_sum = d_g.sum()
        n_total_sum = n_g.sum()
    else:
        d_total_sum = d_g
        n_total_sum = n_g

    # Function to be minimized in weight parameter
    objective_function = lambda params: -calculate_my_likelihood(
        d_total_sum,
        n_total_sum,
        p_g,
        prob_dens_func,
        params[0],
        params[1],
    )

    initial_guess = [w_initial, gamma_initial]

    # Minimization based on the objective function
    result = minimize(objective_function, initial_guess, method='Nelder-Mead', bounds=bounds)

    # The found value of w_g and gamma_g
    w_g_found, gamma_g_found = result.x

    return w_g_found, gamma_g_found


def multivariate_ml_estimation(d_g, n_g, prob_dens_func, p_g, w_initial, gamma_initial, bounds):
    """
    Estimate w_g and gamma_g using the maximum likelihood estimation method.

    Parameters:
        d_g (pd.Series): Time series for d_g.
        n_g (pd.Series): Time series for n_g.
        prob_dens_func (callable): The pdf_g function representing the probability density function.
        p_g (callable): The p_g function representing the probability density function.
        w_initial (float): Initial guess for w_g.
        gamma_initial (list): Initial guess for gamma_g list.
        bounds (list): Bounds for the minimization algorithm.

    Returns:
        tuple: Estimated w_g and gamma_g.
    """

    num_of_grades = len(d_g)

    # Function to be minimized in weight parameter
    objective_function = lambda params: -calculate_likelihood_ts(
        d_g,
        n_g,
        p_g,
        prob_dens_func,
        float(params[0]) * num_of_grades,
        list(params[1:]),
    )

    initial_guess = [w_initial] + gamma_initial

    # Minimization based on the objective function
    result = minimize(objective_function, initial_guess, method='Nelder-Mead', bounds=bounds)

    return result.x


def ml_parameter_estimation(d_g, n_g, p_g, w_initial, gamma_initial, bounds):
    initial_guess = np.array([w_initial, gamma_initial])

    gamma_dim = len(gamma_initial)

    objective_function = lambda params: -np.log(calculate_my_likelihood_arr(
        d_g, n_g, p_g, norm.pdf, params[gamma_dim:], params[0:gamma_dim]))

    result = minimize(objective_function,
                      initial_guess,
                      method="Nelder-Mead",
                      bounds=bounds,
                      options={
                          'disp': True})
    return result
