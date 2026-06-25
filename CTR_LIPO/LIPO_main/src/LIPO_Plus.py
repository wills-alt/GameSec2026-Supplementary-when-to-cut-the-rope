"""
An adaptation of the code of Perceval Beja-Battais, Gaëtan Serré and Sophia Chirrane. This function implements the LIPO+ algorithm

"""

import numpy as np
from collections import deque
from .utils import *


def LIPO_P(f, eps, n: int, window_slope=5, max_slope=600.0):
    """
    Parameters
    ----------
    f: class of the function to maximize (class)
    n: number of function evaluations (int)
    fig_path: path to save the statistics figures (str)
    window_slope: size of the window to compute the slope of the nb_samples vs nb_evaluations curve (int)
    max_slope: maximum slope for the nb_samples vs nb_evaluations curve (float)
    """

    # Initialization
    t = 1
    X_1 = Uniform(f.bounds)
    nb_samples = 1

    # We keep track of the last `window_slope` values of nb_samples to compute the slope
    last_nb_samples = deque([1], maxlen=window_slope)

    points = X_1.reshape(1, -1)
    values = np.array([f(X_1)])

    # Statistics
    stats = []

    # Main loop
    while t < n:
        X_tp1 = Uniform(f.bounds)
        nb_samples += 1
        last_nb_samples[-1] = nb_samples
        if LIPO_condition(X_tp1, values, f.k, points):
            points = np.concatenate((points, X_tp1.reshape(1, -1)))

            values = np.concatenate((values, np.array([f(X_tp1)])))

            # Statistical analysis
            stats.append((np.max(values), nb_samples))

            t += 1
            last_nb_samples.append(0)

        elif slope_stop_condition(last_nb_samples, max_slope):
            print(f"Exponential growth of the number of samples. Stopping the algorithm at iteration {t}." )
            break


    # --- Compute M_hat(T) ---
    M_hat = np.max(values)

    # --- Construct epsilon-near-maximizer set ---
    S_eps = [points[i] for i in range(len(points)) if values[i] > M_hat - eps]

    # Output
    return (M_hat, S_eps)



def PessimisticFollower(T, S_eps, CTRval):
    """
    Compute pessimistic follower response: lambda_eps_star(T) = argmax_{lambda in S_eps} Pr(T, lambda)
    J_eps(T) = max_{lambda in S_eps} Pr(T, lambda)

    Parameters
    ----------
    T : float, Leader action.
    S_eps : list of np.array([lambda]), sampled epsilon-near-maximizers of K(T, lambda).
    Pr_func : function Pr(T, lambda) returning defender loss.

    Returns
    -------
    lambda_eps_star : Worst-case lambda in S_eps for the defender (float).
    J_eps : Worst-case defender loss over S_eps (float).
    """

    # Evaluate Pr(T, lambda) for each lambda in S_eps
    Pr_values = []
    for lam_arr in S_eps:
        lam = float(lam_arr)
        Pr_values.append(CTRval(T, lam))   

    idx = int(np.argmax(Pr_values))
    lambda_eps_star = float(S_eps[idx])
    J_eps = Pr_values[idx]

    return lambda_eps_star, J_eps
