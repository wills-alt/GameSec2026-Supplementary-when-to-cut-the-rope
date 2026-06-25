"""
Code developed by Perceval Beja-Battais, Gaëtan Serré and Sophia Chirrane
"""

import numpy as np


def Uniform(X: np.array):
    """
    This function generates a random point in the feasible region X. We assume that X is a subset of R^n
    described by the inequalities X = {x in R^n | a_i <= x_i <= b_i, i = 0, ..., m-1} where a_i, b_i are given
    such that X[i,j] = [a_i, b_i] for i = 0, ..., m-1 and j = 0, 1.
    For simplicity, we assume that X C Rectangle given by an infinite norm (i.e. X = {x in R^n | -M <= x_i <= M, i = 1, ..., n}).
    X: feasible region (numpy array)
    """

    theta = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        theta[i] = np.random.uniform(X[i, 0], X[i, 1])
    return theta


def Bernoulli(p: float):
    """
    This function generates a random variable following a Bernoulli distribution.
    p: probability of success (float)
    """
    a = np.random.uniform(0, 1)
    if a <= p:
        return 1
    else:
        return 0


def slope_stop_condition(last_nb_samples, max_slope):
    """
    Check if the slope of the last points of the the nb_samples vs nb_evaluations curve
    is greater than max_slope.
    """
    slope = (last_nb_samples[-1] - last_nb_samples[0]) / (len(last_nb_samples) - 1)
    return slope > max_slope


def LIPO_condition(x, values, k, points):
    """
    Subfunction to check the condition in the loop, depending on the set of values we already have.
    values: set of values of the function we explored (numpy array)
    x: point to check (numpy array)
    k: Lipschitz constant (float)
    points: set of points we have explored (numpy array)
    """
    max_val = np.max(values)

    left_min = np.min(
        values.reshape(-1) + k * np.linalg.norm(x - points, ord=2, axis=1)
    )

    return left_min >= max_val
