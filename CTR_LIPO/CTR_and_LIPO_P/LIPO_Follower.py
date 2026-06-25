from CTR_LIPO.LIPO_main.src.LIPO_Plus import LIPO_P, PessimisticFollower
from CTR_LIPO.CTR.CTR_eval_Poisson import run_experiment

from skopt import gp_minimize
from skopt.space import Real
import numpy as np

from ..graphs.attack_graph_MARA import create_mara_attack_graph
from ..graphs.attack_graph_MARA import mara_set_default_weight
from ..graphs.attack_graph_MARA import plot_main_graph

from ..graphs.attack_graph_MIR100 import *
from ..graphs.attack_graph_UNGUARD import *


# Follower Lipschitz constant in λ:  |K(T, λ2) - K(T, λ1)| ≤ (T_max + ω β) |λ2 - λ1|

def L_lambda(T_max, omega, beta):
    return T_max + omega * beta

# Follower cost function (with β = 0.1)
def c_a(lam):
    return 9 * (lam**2)/2

# Leader cost function (with α = 0.2)
def c_d(T):
    return (400/T) + T - 1


# Optimal probability of the attacker obtained from launching the CTR game: Make sure to choose the correct graph that you want to experiment
#create_mara_attack_graph, create_mir100_attack_graph or create_UNGUARD_attack_graph

def CTRval(T, lam):
    result = run_experiment(attack_rate=[lam], defense_rate=[T], graph_considered = create_UNGUARD_attack_graph)
    return result["all_equilibria"][(T, lam)]["attacker_success"]

def get_graph_name(graph_considered=create_UNGUARD_attack_graph):
    return graph_considered.__name__


class FollowerObjective:
    def __init__(self, T, lambda_min, lambda_max, L_lambda, CTRval, c_a, omega, T_max, beta, eps):
        self.T = T
        self.CTRval = CTRval
        self.L_lambda = L_lambda
        self.c_a = c_a
        self.omega = omega
        self.bounds = np.array([[lambda_min, lambda_max]]) 
        self.T_max =T_max
        self.beta =beta
        self.eps = eps
        self.k = self.L_lambda(T_max, omega, beta)                              

    def __call__(self, x):
        lam = float(x[0])
        val = self.CTRval(self.T, lam)   
        return val - self.omega * self.c_a(lam)



def lipofollower(T, lambda_min, lambda_max, L_lambda, n_lambda, omega, T_max, beta, eps, CTRval):
    """
    LIPOFollower: approximate the follower best response λ*(T) of a leader action T via CTR and LIPO+.

    Parameters
    ----------
    T : A fixed leader decision.
    lambda_min, lambda_max (float): Bounds [λ_min, λ_max] of the follower domain.
    L_lambda_T (float): Lipschitz constant L_λ(T).
    n_lambda (int): Safety budget (max number of accepted evaluations, depending on the accuracy wanted).
    omega (float): Weight in the follower payoff funtion K(T, λ) = Pr(T, λ) - ω c_a(λ).    
    """

    # Build the LIPO_P-compatible objective
    K_obj = FollowerObjective(T=T, lambda_min=lambda_min, lambda_max=lambda_max, 
                                L_lambda=L_lambda, CTRval=CTRval, c_a=c_a, omega=omega, T_max = T_max, beta = beta, eps = eps)

    # Run LIPO_P
    M_hat, S_eps = LIPO_P(f = K_obj, eps = eps, n = n_lambda, window_slope = 5, max_slope = 600.0)

    # Extract best λ and the associated J_epsilon, the maximum attacker probability success
    lambda_eps_star, J_eps = PessimisticFollower(T, S_eps, CTRval)

    return lambda_eps_star, J_eps




def Eps_Pessimistic_Stackelberg(T_min, T_max, lambda_min, lambda_max, L_lambda, n_lambda, omega, beta, eps, gamma, n_calls=30):
    """
    Eps-Pessimistic-Stackelberg: Compute (T_eps_star, lambda_eps_star(T_eps_star)) via epsilon-regularized pessimistic Stackelberg equilibrium.
    Parameters
    ----------
    lambda_min, lambda_max : Bounds of lambda.
    L_lambda : Lipschitz constant function L_lambda(T_max, omega, beta).
    n_lambda :  Follower LIPO budget.
    omega, T_max, beta :  Model parameters.
    eps :  Epsilon for near-maximizers.
    n_calls : Number of evaluations of J_tilde(T).

    Returns
    -------
    T_eps_star : float
    lambda_eps_star : float
    """


    # Objective for the leader: J_tilde(T) = J_eps(T) + c_d(T)
    def J_tilde(T_candidate):
        lambda_eps_star, J_eps = lipofollower(T_candidate, lambda_min, lambda_max, L_lambda, n_lambda, omega, T_max, beta, eps, CTRval)
        return J_eps + gamma * c_d(T_candidate)

    # Bayesian optimization over T in [T_min, T_max]
    space = [Real(T_min, T_max, name="T")]

    res = gp_minimize(func=lambda x: J_tilde(x[0]), dimensions=space, n_calls=n_calls, random_state=0)

    T_eps_star = res.x[0]

    # Recompute follower at T_eps_star to get lambda_eps_star
    lambda_eps_star, J_eps_star = lipofollower(T_eps_star, lambda_min, lambda_max, L_lambda, n_lambda, omega, T_max, beta, eps, CTRval)

  
    return T_eps_star, lambda_eps_star

