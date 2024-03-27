import numpy as np
import random
from scipy.optimize import minimize
from scipy import stats


def expected_value_of_function_monte_carlo(f, num_samples=100000):
    """
    Calculates the expected value of a given function f(x) where x follows the standard normal distribution.

    Parameters:
        f (callable): The function f(x) to calculate the expected value.
        num_samples (int): The number of samples to use in the Monte Carlo simulation.

    Returns:
        float: The estimated expected value of the function.
    """
    # Generate random samples from the standard normal distribution
    random_samples = np.random.normal(size=num_samples)

    # Calculate the expected value using the Monte Carlo simulation
    estimated_expected_value = np.mean(f(random_samples))

    return estimated_expected_value



def likelihood_mc(w_g, gamma_g, d_g, n_g, p_g, num_of_simulations):
    # likelihood function
    likelihood_function = lambda x: np.prod([stats.binom.pmf(d, n, p_g(x, w_g=w_g, gamma_g=gamma)) for d, n, gamma in zip(d_g, n_g, gamma_g)], axis=0)

    return expected_value_of_function_monte_carlo(likelihood_function, num_of_simulations)

def monte_carlo_MLE(d_g, n_g, p_g, w_initial, gamma_initial, bounds, num_of_simulations, seed=None):
    """
    Estimate w_g and gamma_g using the maximum likelihood estimation method.
    Parameters:
        d_g (list): number of defaults.
        n_g (list): number of obligors.
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
        seed = random.randint(1, 10000)

    np.random.seed(seed)

    # objective function
    objective_function = lambda params: -np.log(likelihood_mc(params[0], params[1:], d_g, n_g, p_g, num_of_simulations))

    initial_guess = w_initial + gamma_initial

    result = minimize(objective_function, initial_guess, method='Nelder-Mead', bounds=bounds)

    # The found value of w_g and gamma_g
    return result
