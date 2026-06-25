import numpy as np

from CTR_LIPO.CTR_and_LIPO_P.LIPO_Follower import *
import pandas as pd

# ------------------ PARAMETERS ------------------

T_min, T_max = 2, 80
lambda_min, lambda_max = 0.1, 0.8

beta  = 9 * lambda_max
gamma = 0.07
omega = 0.015

n_calls = 600
eps = 0.0000000000002
n_lambda = 300




# ------------------ GAMMA AND OMEGA VALUES ------------------

gamma_values = [0.01, 0.2, 0.9, 1.5, 3.5]
omega_values = [0.02, 0.2, 0.5]
results = []


for gamma in gamma_values:
    for omega in omega_values:

        # Compute Stackelberg equilibrium
        T_eps_star, lambda_eps_star = Eps_Pessimistic_Stackelberg(
                    T_min = T_min, T_max = T_max,
                    lambda_min =lambda_min,
                    lambda_max =lambda_max,
                    L_lambda =L_lambda,
                    n_lambda = n_lambda,
                    omega =omega,
                    beta = beta,
                    eps = eps,
                    gamma = gamma,
                    n_calls = n_calls)

        # Success probability at equilibrium
        P_T_eps_star = CTRval(T_eps_star, lambda_eps_star)

        # Defender utility
        J_star = P_T_eps_star + gamma * c_d(T_eps_star)
        K_star = P_T_eps_star - omega * c_a(lambda_eps_star)

        results.append((gamma, omega, T_eps_star, lambda_eps_star, P_T_eps_star, J_star, K_star))


    # ------------------ PRINT TABLE ------------------


print("\nStackelberg Equilibrium Table\n")
print(f"{'gamma':>8} | {'omega':>8} | {'T*':>8} | {'λ*':>8} | {'P_T*':>10} | {'J(T*)':>10} | {'K(λ*)':>10}")
print("-" * 46)

for gamma, omega, T_eps_star, lambda_eps_star, P_T_eps_star, J_star, K_star in results:
    print(f"{gamma:8.2f} | {omega:8.2f} | {T_eps_star:8.3f} | {lambda_eps_star:8.3f} | {P_T_eps_star:10.4f} | {J_star:10.4f} | {K_star:10.4f}")

print("\nLaTeX-ready format:\n")

print("\\begin{tabular}{cccc}")
print("\\toprule")
print("$\gamma$, $\omega$, & $T^*$, $\lambda^*$, & $P_{T^*}$ & $J(T^*)$, $K(\lambda^*)$\\\\")
print("\\midrule")

for gamma, omega, T_eps_star, lambda_eps_star, P_T_eps_star, J_star, K_star in results:
    print(f"{gamma:.2f} & {omega:.2f} &  {T_eps_star:.3f} & {lambda_eps_star:.3f} & {P_T_eps_star:.4f} & {J_star:.4f} & {K_star:.4f} \\\\")
    
print("\\bottomrule")
print("\\end{tabular}")

df = pd.DataFrame(results, columns=[
    "gamma",
    "omega",
    "T_star",
    "lambda_star",
    "P_T_star",
    "J_star",
    "K_star"
])

#create_mir100_attack_graph, create_mara_attack_graph, create_UNGUARD_attack_graph

graph_name = get_graph_name(create_mara_attack_graph)
csv_path = fr"CSV_file_saving_direction/ML_grid_results_{graph_name}.csv"
#df.to_csv(csv_path, index=False)

#print(f"\n Results saved to: {csv_path}\n")
