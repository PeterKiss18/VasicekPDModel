import numpy as np
from scipy.optimize import minimize
from scipy import stats
from scipy.stats import norm


def calc_variance_of_default_rate(w_factor_loading, pd_average):
    """
    Calculate the variance of the default rate
    :param w_factor_loading: float, factor loading
    :param pd_average: float, average of the default rate
    :return: float, variance of the default rate
    """
    cut_off_value = stats.norm.ppf(pd_average)

    # if w_factor_loading is a np array, take the first element
    if isinstance(w_factor_loading, np.ndarray):
        w_factor_loading = w_factor_loading[0]

    # Bivariate normal cdf parameters
    mean = [0, 0]
    cov_matrix = [[1, w_factor_loading], [w_factor_loading, 1]]

    # Calculate BIVNOR value
    bivnor_value = stats.multivariate_normal.cdf([cut_off_value, cut_off_value], mean=mean, cov=cov_matrix)

    result = bivnor_value - pd_average ** 2

    return result


def estimate_w_factor_loading(historical_pd, num_of_total_grades, initial_guess=0.27):
    """
    Estimate w_factor_loading using the method of moments
    :param historical_pd: pd series, list of historical default rates
    :param num_of_total_grades: pd series, list of the number of total grades
    :param initial_guess: float, initial guess for the minimization algorithm
    :return: float, estimated w_factor_loading
    """

    # Calculate the average
    pd_average = np.mean(historical_pd)

    # Calculate the expected value of 1/n_g
    expected_value_of_reciprocal_n_g = np.mean(1 / num_of_total_grades)

    # Variance of default rate
    variance_of_p_d = (np.var(historical_pd) - expected_value_of_reciprocal_n_g * pd_average * (1 - pd_average)) / (
                1 - expected_value_of_reciprocal_n_g)

    # Define bounds for w_factor_loading
    bounds = [(-0.9999, 0.9999)]

    # Define object function
    object_function = lambda w: abs(calc_variance_of_default_rate(w, pd_average) - variance_of_p_d)

    # Minimization based on the objective function
    result = minimize(object_function, initial_guess, bounds=bounds)

    # The found value of w_factor_loading
    w_factor_loading_found = result.x[0]

    return w_factor_loading_found, pd_average


def MM_estimation(default_series, num_of_obligors_series, w_init=0.27):
    """
    Estimate w_g and gamma_g using the MM estimation method.

    Returns:
        tuple: Estimated w_g and gamma_g.
    """
    historical_pd = default_series / num_of_obligors_series

    # Estimate w_factor_loading
    w_g, pd_average = estimate_w_factor_loading(historical_pd, num_of_obligors_series, w_init)

    return w_g, norm.ppf(pd_average)