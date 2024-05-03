import numpy as np
import pandas as pd
from scipy.stats import binom, norm
from scipy.integrate import quad
from scipy.optimize import minimize
from src.data_generator import generate_default_buckets
from src.sucess_probability import p_g
from src.variable_change import w_calc_func, gamma_calc_func, a_calc_func, b_calc_func
from scipy.integrate import cumtrapz
from src.data_generator import generate_default_time_series


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

    result, _ = quad(integrand, -3, 3, epsabs=1.49e-28)

    return result


def calculate_trapz_likelihood_arr(d_g_arr, n_g_arr, p_g, prob_dens_func, w_g_arr, gamma_g_arr):
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

    # Generate a range of values for y from 0 to 1
    y_values = np.linspace(0, 1, num=1000)

    # Calculate the integrand at each value of y
    integrand_values = np.prod(binom.pmf(d_g_arr, n_g_arr, p_g(norm.ppf(y_values), w_g_arr, gamma_g_arr)), axis=1)

    # Integrate the integrand using cumtrapz
    result = cumtrapz(integrand_values, y_values)[-1]

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


def parameter_estimation(default_list, num_of_obligors_over_time, factor_loading_init, gamma_list_init):
    initial_guess = gamma_list_init + factor_loading_init

    num_of_gamma = len(gamma_list_init)
    num_of_factor_loading = len(factor_loading_init)
    bounds = num_of_gamma * [(-5, 5)] + num_of_factor_loading * [(-1, 1)]
    # Optimization
    objective_function = lambda params: -np.log(calculate_my_likelihood_arr(
        default_list, num_of_obligors_over_time, p_g, norm.pdf,
        params[num_of_gamma:num_of_gamma+num_of_factor_loading], params[0:num_of_gamma]
    ))

    result = minimize(objective_function,
                      initial_guess,
                      method="Nelder-Mead",
                      bounds=bounds,
                      options={
                          'disp': False})

    return result


def gen_data_and_mle(time_points, num_of_obligors_list, factor_loading_list, gamma_list, factor_loading_init, gamma_list_init):
    d_g_list = generate_default_buckets(factor_loading_list, num_of_obligors_list, gamma_list, time_points)
    n_g_list = [num_of_obligors_list[i] * time_points for i in range(len(num_of_obligors_list))]

    ml_params = parameter_estimation(d_g_list, n_g_list, factor_loading_init, gamma_list_init)

    return d_g_list, n_g_list, ml_params.x


def calculate_variable_changed_likelihood_arr(d_g_arr, n_g_arr, p_g, prob_dens_func, a, b):
    integrand = lambda x: np.prod(binom.pmf(d_g_arr, n_g_arr, norm.cdf(a*x+b))) * prob_dens_func(x)

    result, _ = quad(integrand, -3, 3, epsabs=1.49e-28)

    return result


def ml_estimation_linear(default_list, num_of_obligors_over_time, a_init, b_init):
    initial_guess = np.concatenate((a_init, b_init))

    num_of_a = len(a_init)
    bounds = [(-10, 10)] * len(initial_guess)

    # Optimization
    objective_function = lambda params: -np.log(calculate_variable_changed_likelihood_arr(
        default_list, num_of_obligors_over_time, p_g, norm.pdf, params[:num_of_a], params[num_of_a:len(initial_guess)]
    ))

    result = minimize(objective_function,
                      initial_guess,
                      method="Nelder-Mead",
                      bounds=bounds,
                      options={
                          'disp': False})

    return result


def ml_estimation_linear_with_w_and_g(
        default_list, num_of_obligors_over_time, factor_loading_init, gamma_list_init, fixed_w=False, fixed_g=False):
    # if len(factor_loading_init) == 1:
    #     factor_loading_init = np.full_like(gamma_list_init, factor_loading_init[0])
    # elif len(gamma_list_init) == 1:
    #     gamma_list_init = gamma_list_init * len(factor_loading_init)

    a_init = np.array(a_calc_func(np.array(factor_loading_init), np.array(gamma_list_init)))
    b_init = np.array(b_calc_func(np.array(factor_loading_init), np.array(gamma_list_init)))

    initial_guess = np.concatenate((a_init, b_init))

    num_of_a = len(a_init)
    bounds = [(-10, 10)] * len(initial_guess)

    # Optimization
    if not fixed_w and not fixed_g:
        objective_function = lambda params: -np.log(calculate_variable_changed_likelihood_arr(
            default_list, num_of_obligors_over_time, p_g, norm.pdf, params[:num_of_a], params[num_of_a:len(initial_guess)]
        ))

        result = minimize(objective_function,
                          initial_guess,
                          method="Nelder-Mead",
                          bounds=bounds,
                          options={
                              'disp': False})

        factor_loading_result = np.array(w_calc_func(np.array(result.x[:num_of_a]), np.array(result.x[num_of_a:])))
        gamma_result = np.array(gamma_calc_func(np.array(result.x[:num_of_a]), np.array(result.x[num_of_a:])))

    elif fixed_w:
        objective_function = lambda params: -np.log(calculate_variable_changed_likelihood_arr(
            default_list, num_of_obligors_over_time, p_g, norm.pdf, a_init, params
        ))

        result = minimize(objective_function,
                          b_init,
                          method="Nelder-Mead",
                          bounds=bounds[num_of_a:],
                          options={
                              'disp': False})

        factor_loading_result = np.array(w_calc_func(a_init, result.x))
        gamma_result = np.array(gamma_calc_func(a_init, result.x))

    elif fixed_g:
        objective_function = lambda params: -np.log(calculate_variable_changed_likelihood_arr(
            default_list, num_of_obligors_over_time, p_g, norm.pdf, params, b_init
        ))

        result = minimize(objective_function,
                          a_init,
                          method="Nelder-Mead",
                          bounds=bounds[:num_of_a],
                          options={
                              'disp': False})

        factor_loading_result = np.array(w_calc_func(result.x, b_init))
        gamma_result = np.array(gamma_calc_func(result.x, b_init))

    return factor_loading_result, gamma_result, result

def calc_linear_likelihood(d_g_arr, n_g_arr, p_g, prob_dens_func, a, b):
    y_values = np.linspace(0, 1, num=1000)

    y_dim = len(y_values)
    a_dim = len(a)

    a_mat = np.tile(a, (y_dim, 1))
    b_mat = np.tile(b, (y_dim, 1))
    y_mat = np.tile(y_values, (a_dim, 1)).T

    integrand_values = np.prod(binom.pmf(d_g_arr, n_g_arr, norm.cdf(a_mat * norm.ppf(y_mat) + b_mat)), axis=1)
    result = cumtrapz(integrand_values, y_values)[-1]
    return result


def log_likehood_variable_changed_fast(d_g_array, n_g_array, p_g, prob_dens_func, a, b):
    return sum(np.log(calc_linear_likelihood(d_g_list, n_g_list, p_g, prob_dens_func, a, b)) for d_g_list, n_g_list in zip(d_g_array, n_g_array))


def mle_linear_trapz(default_table, num_of_obligors_table, a_init, b_init, fixed_a=False):
    a_init = np.asarray([a_init]) if isinstance(a_init, float) else np.asarray(a_init)
    b_init = np.asarray([b_init]) if isinstance(b_init, float) else np.asarray(b_init)

    if not fixed_a:
        initial_guess = np.concatenate((a_init, b_init))

        num_of_a = len(a_init)
        bounds = [(-10, 10)] * len(initial_guess)

        # Optimization
        objective_function = lambda params: -log_likehood_variable_changed_fast(
            default_table, num_of_obligors_table, p_g, norm.pdf, params[:num_of_a], params[num_of_a:len(initial_guess)]
        )

        result = minimize(objective_function,
                          initial_guess,
                          method="Nelder-Mead",
                          bounds=bounds,
                          options={
                              'disp': False})

    else:
        initial_guess = b_init

        bounds = [(-10, 10)] * len(initial_guess)

        # Optimization
        objective_function = lambda params: -log_likehood_variable_changed_fast(
            default_table, num_of_obligors_table, p_g, norm.pdf, a_init, params
        )

        result = minimize(objective_function,
                          initial_guess,
                          method="Nelder-Mead",
                          bounds=bounds,
                          options={
                              'disp': False})

    return result

def mle_trapz_g_and_w(
        default_table, num_of_obligors_table, factor_loading_init, gamma_list_init, fixed_w=False, fixed_g=False):
    # if len(factor_loading_init) == 1:
    #     factor_loading_init = np.full_like(gamma_list_init, factor_loading_init[0])
    # elif len(gamma_list_init) == 1:
    #     gamma_list_init = gamma_list_init * len(factor_loading_init)

    a_init = np.array(a_calc_func(np.array(factor_loading_init), np.array(gamma_list_init)))
    b_init = np.array(b_calc_func(np.array(factor_loading_init), np.array(gamma_list_init)))

    initial_guess = np.concatenate((a_init, b_init))

    num_of_a = len(a_init)
    bounds = [(-10, 10)] * len(initial_guess)

    # Optimization
    if not fixed_w and not fixed_g:
        objective_function = lambda params: -log_likehood_variable_changed_fast(
            default_table, num_of_obligors_table, p_g, norm.pdf, params[:num_of_a], params[num_of_a:len(initial_guess)]
        )

        result = minimize(objective_function,
                          initial_guess,
                          method="Nelder-Mead",
                          bounds=bounds,
                          options={
                              'disp': False})

        factor_loading_result = np.array(w_calc_func(np.array(result.x[:num_of_a]), np.array(result.x[num_of_a:])))
        gamma_result = np.array(gamma_calc_func(np.array(result.x[:num_of_a]), np.array(result.x[num_of_a:])))

    elif fixed_w:
        objective_function = lambda params: -log_likehood_variable_changed_fast(
            default_table, num_of_obligors_table, p_g, norm.pdf, a_init, params
        )

        result = minimize(objective_function,
                          b_init,
                          method="Nelder-Mead",
                          bounds=bounds[num_of_a:],
                          options={
                              'disp': False})

        factor_loading_result = np.array(w_calc_func(a_init, result.x))
        gamma_result = np.array(gamma_calc_func(a_init, result.x))

    elif fixed_g:
        objective_function = lambda params: -log_likehood_variable_changed_fast(
            default_table, num_of_obligors_table, p_g, norm.pdf, params, b_init
        )

        result = minimize(objective_function,
                          a_init,
                          method="Nelder-Mead",
                          bounds=bounds[:num_of_a],
                          options={
                              'disp': False})

        factor_loading_result = np.array(w_calc_func(result.x, b_init))
        gamma_result = np.array(gamma_calc_func(result.x, b_init))

    return factor_loading_result, gamma_result, result

def gen_data_and_ml_estimation(time_points, num_of_obligors_list, factor_loading_list, gamma_list, sims=100):
    """
    Generate data and maximum likelihood estimation.
    :param time_points: int, number of time points
    :param num_of_obligors_list: list, number of obligors for each grade
    :param factor_loading_list: list, factor loading for each grade
    :param gamma_list: list, gamma for each grade
    :param sims: int, number of simulations
    """
    params_df = pd.DataFrame()

    for sim in range(sims):
        defaults_df = generate_default_time_series(factor_loading_list, num_of_obligors_list, gamma_list, time_points)
        num_of_obligors_df = np.array([num_of_obligors_list] * len(defaults_df))
        w_param, pd_param, _ = mle_trapz_g_and_w(defaults_df.values, num_of_obligors_df, factor_loading_list, gamma_list)

        for i in range(len(factor_loading_list)):
            params_df.loc[sim, "w_" + str(i)] = w_param[i]

        for i in range(len(gamma_list)):
            params_df.loc[sim, "gamma_" + str(i)] = pd_param[i]

    return params_df


def gen_data_and_mle1(time_points, num_of_obligors_list, factor_loading_list, gamma_list, sims=100):
    """
    Generate data and estimate parameters using the MLE1 method, which is MLE by marginals.
    :param time_points: int, number of time points
    :param num_of_obligors_list: list, number of obligors for each grade
    :param factor_loading_list: list, factor loading for each grade
    :param gamma_list: list, gamma for each grade
    :param sims: int, number of simulations
    """
    params_df = pd.DataFrame()

    for sim in range(sims):
        defaults_df = generate_default_time_series(factor_loading_list, num_of_obligors_list, gamma_list, time_points)

        for col in range(defaults_df.shape[1]):
            defaults_col = defaults_df.iloc[:, col]
            num_of_obligors_col = np.array([num_of_obligors_list[col]] * len(defaults_col))
            w_param, pd_param, _ = mle_trapz_g_and_w(defaults_col.values,
                                                     num_of_obligors_col,
                                                     [factor_loading_list[col]],
                                                     [gamma_list[col]])

            params_df.loc[sim, f"w_{col}"] = w_param[0]
            params_df.loc[sim, f"gamma_{col}"] = pd_param[0]

    return params_df
