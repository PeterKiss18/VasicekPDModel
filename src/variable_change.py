import numpy as np


def w_calc_func(a, b):
    return -a / np.sqrt(a**2 + 1)


def gamma_calc_func(a, b):
    return b * np.sqrt(1 - w_calc_func(a, b)**2)


def a_calc_func(w, gamma):
    return - w / np.sqrt(1 - w**2)


def b_calc_func(w, gamma):
    return gamma / np.sqrt(1 - w**2)
