import numpy as np
import matplotlib.pyplot as plt
from CTR_LIPO.CTR_and_LIPO_P.LIPO_Follower import *
import warnings
warnings.filterwarnings(
    "ignore",
    message="starts with '_'. It is thus excluded from the legend.",
    category=UserWarning
)

import pandas as pd

# ---------------------------------------------------------
# PARAMETERS
# ---------------------------------------------------------

T_min, T_max = 2, 80
lambda_min, lambda_max = 0.1, 0.8

beta  = 9 * lambda_max
gamma = 0.07
omega = 0.015
n_calls = 20
eps = 0.0000000000002

n_lambda = 2  

# ---------------------------------------------------------
# COMPUTE STACKELBERG EQUILIBRIUM
# ---------------------------------------------------------

T_star, lambda_star = Eps_Pessimistic_Stackelberg(
    T_min = T_min, T_max = T_max,
    lambda_min = lambda_min,
    lambda_max = lambda_max,
    L_lambda = L_lambda,
    n_lambda = n_lambda,
    omega = omega,
    beta = beta,
    eps = eps,
    gamma = gamma,
    n_calls = n_calls
)

print("Stackelberg equilibrium:")
print("  T* =", T_star)
print("  λ* =", lambda_star)

# ---------------------------------------------------------
# DEFENDER DEVIATION CURVE
# ---------------------------------------------------------

T_grid = np.linspace(T_min, T_max, 3)
T_grid = np.append(T_grid, T_star)

J_defender_dev = []

for T in T_grid:
    # follower best response to T
    lam_br, J_eps = lipofollower(
        T=T,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        L_lambda=L_lambda,
        n_lambda=n_lambda,
        omega=omega,
        T_max=T_max,
        beta=beta,
        eps=eps,
        CTRval=CTRval
    )
    # Stackelberg leader payoff
    J_defender_dev.append(J_eps + gamma * c_d(T))

T_grid, J_defender_dev = zip(*sorted(zip(T_grid, J_defender_dev)))
T_grid = np.array(T_grid)
J_defender_dev = np.array(J_defender_dev)

# ---------------------------------------------------------
# ATTACKER DEVIATION CURVE
# ---------------------------------------------------------

lambda_grid = np.linspace(lambda_min, lambda_max, 3)
lambda_grid = np.append(lambda_grid, lambda_star)

K_attacker_dev = [
    CTRval(T_star, lam) - omega * c_a(lam)
    for lam in lambda_grid
]

lambda_grid, K_attacker_dev = zip(*sorted(zip(lambda_grid, K_attacker_dev)))
lambda_grid = np.array(lambda_grid)
K_attacker_dev = np.array(K_attacker_dev)

# ---------------------------------------------------------
# PLOT
# ---------------------------------------------------------

fig, ax1 = plt.subplots(figsize=(12, 6))

# Defender curve
ax1.plot(T_grid, J_defender_dev, color="blue", label="Defender payoff variation")
ax1.axvline(T_star, color="blue", linestyle="--", linewidth=1.5)
ax1.set_xlabel("Defender timing T")
ax1.set_ylabel(r"$\tilde{J}(T,\lambda^*(T))$", color="blue")
ax1.tick_params(axis='y', labelcolor="blue")
ax1.grid(True, linestyle="--", alpha=0.6)

# Attacker curve on twin axes
ax2 = ax1.twiny()
ax3 = ax1.twinx()

lam_to_T = lambda lam: T_min + (T_max - T_min) * (lam - lambda_min) / (lambda_max - lambda_min)
T_for_lambda = lam_to_T(lambda_grid)

ax3.plot(T_for_lambda, K_attacker_dev, color="red", label="Attacker payoff variation")
ax3.axvline(lam_to_T(lambda_star), color="red", linestyle="--", linewidth=1.5)
ax3.set_ylabel(r"$K(T^*,\lambda)$", color="red")
ax3.tick_params(axis='y', labelcolor="red")

# Top axis
ax2.set_xlim(ax1.get_xlim())
step = max(1, len(lambda_grid)//6)
ax2.set_xticks(T_for_lambda[::step])
ax2.set_xticklabels([f"{lam:.1f}" for lam in lambda_grid[::step]])
ax2.set_xlabel("Attacker rate λ")

# Labels for vertical lines
ax1.text(T_star + 0.3, np.mean(J_defender_dev), r"$T^*$",
         color="blue", fontsize=12, fontweight="bold", va="center")

ax3.text(lam_to_T(lambda_star) + 0.3, np.mean(K_attacker_dev), r"$\lambda^*$",
         color="red", fontsize=12, fontweight="bold", va="center")

plt.title("Optimal strategies assessment for defender and attacker")

# Legend
lines, labels = [], []
for ax in [ax1, ax3]:
    for line in ax.get_lines():
        label = line.get_label()
        if label and not label.startswith("_"):
            lines.append(line)
            labels.append(label)

plt.legend(lines, labels, loc="best")
plt.tight_layout()

#create_mara_attack_graph, create_mir100_attack_graph, create_UNGUARD_attack_graph

graph_name = get_graph_name(create_mara_attack_graph)

# Modify and insert the correct directory

# plt.savefig(
#     fr'Plots_Saving_direction/Optimal_vs_suboptimal_SE_{graph_name}.png',
#     bbox_inches="tight", dpi=500
# )

# ---------------------------------------------------------
# SAVE RESULTS TO CSV
# ---------------------------------------------------------

max_len = max(len(T_grid), len(lambda_grid))

df = pd.DataFrame({
    "T_grid": np.pad(T_grid, (0, max_len - len(T_grid)), constant_values=np.nan),
    "J_defender_dev": np.pad(J_defender_dev, (0, max_len - len(J_defender_dev)), constant_values=np.nan),
    "lambda_grid": np.pad(lambda_grid, (0, max_len - len(lambda_grid)), constant_values=np.nan),
    "K_attacker_dev": np.pad(K_attacker_dev, (0, max_len - len(K_attacker_dev)), constant_values=np.nan),
})

df["T_star"] = T_star
df["lambda_star"] = lambda_star

# csv_path = fr"CSV_file_saving_direction/SE_results_{graph_name}.csv"
# df.to_csv(csv_path, index=False)

#print(f"\n Results saved to: {csv_path}\n")
plt.show()
