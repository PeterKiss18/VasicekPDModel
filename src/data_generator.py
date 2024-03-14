import numpy as np


def generate_y(factor_loading, num_of_obligors):
    x = np.random.normal()
    epsilon = np.random.normal(0, 1, num_of_obligors)
    y = factor_loading * x + epsilon * (1 - factor_loading**2)**0.5
    return y


def generate_default(num_of_obligors, factor_loading, gamma):
    y = generate_y(factor_loading, num_of_obligors)
    default = (y < gamma).sum()
    return default


def generate_default_buckets(factor_loading_list, num_of_obligors_list, gamma_list, time_points = 160):
    x = np.random.normal(0, 1, time_points)
    defaults_list = []

    for index, num_of_obligors in enumerate(num_of_obligors_list):
        d_g = 0
        for i in range(time_points):
            epsilon = np.random.normal(0, 1, num_of_obligors)
            y = factor_loading_list[index] * x[i] + epsilon * (1 - factor_loading_list[index] ** 2) ** 0.5
            d_g += (y < gamma_list[index]).sum()
        defaults_list.append(d_g)

    return defaults_list
