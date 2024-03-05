import numpy as np
import pandas as pd
import random
from scipy.optimize import minimize
from scipy import stats


def monte_carlo_MLE(d_g, n_g, p_g, w_initial, gamma_initial, bounds, num_of_simulations, seed=None):
    """
    Estimate w_g and gamma_g using the maximum likelihood estimation method.
    Parameters:
        d_g (pd.Series): Time series for d_g.
        n_g (pd.Series): Time series for n_g.
        p_g (callable): The p_g function representing the probability density function.
        w_initial (float): Initial guess for w_g.
        gamma_initial (float): Initial guess for gamma_g.
        bounds (list): Bounds for the minimization algorithm.
        num_of_simulations (int): Number of simulations for the Monte Carlo method.
        seed (int): Seed for the random number generator.
    Returns:
        tuple: Estimated w_g and gamma_g.
    """
    if seed is None:
        seed = random.randint(1, 1000)

    random.seed(seed)

    # Sum of d_g and n_g if they are pd.Series
    if isinstance(d_g, pd.Series):
        d_total_sum = d_g.sum()
        n_total_sum = n_g.sum()
    else:
        d_total_sum = d_g
        n_total_sum = n_g

    random_samples = np.random.normal(size=num_of_simulations)

    # likelihood function
    likelihood_func = lambda x, w, gamma: stats.binom.pmf(d_total_sum, n_total_sum, p_g(x, w_g=w, gamma_g=gamma))

    # objective function
    objective_function = lambda params: -np.mean(likelihood_func(random_samples, *params))

    initial_guess = [w_initial, gamma_initial]

    result = minimize(objective_function, initial_guess, method='Nelder-Mead', bounds=bounds)

    # The found value of w_g and gamma_g
    return result.x[0], result.x[1]
