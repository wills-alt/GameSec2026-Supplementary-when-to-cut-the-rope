import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# GLOBAL STYLE 
# ---------------------------------------------------------
plt.rcParams.update({
    "font.size": 22,           
    "axes.titlesize": 26,       
    "axes.labelsize": 24,     
    "xtick.labelsize": 20,      
    "ytick.labelsize": 20,
    "legend.fontsize": 22,
    "lines.linewidth": 3,       
    "axes.linewidth": 2,        
})

# ---------------------------------------------------------
# CONFIGURATIONS
# ---------------------------------------------------------

csv_folder = "Location where results have been saved"

graphs = [
    "create_mara_attack_graph",
    "create_mir100_attack_graph",
    "create_UNGUARD_attack_graph"
]


# ---------------------------------------------------------
# FUNCTION TO GENERATE ONE FIGURE
# ---------------------------------------------------------

def plot_single_graph(graph_name):

    df = pd.read_csv(f"{csv_folder}/SE_results_{graph_name}.csv")

    T_grid = df["T_grid"].dropna().values
    J_def = df["J_defender_dev"].dropna().values
    lambda_grid = df["lambda_grid"].dropna().values
    K_att = df["K_attacker_dev"].dropna().values

    T_star = df["T_star"].dropna().iloc[0]
    lambda_star = df["lambda_star"].dropna().iloc[0]


    fig, ax1 = plt.subplots(figsize=(16, 10))

    # Defender curve
    ax1.plot(T_grid, J_def, color="blue")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.set_xlabel("Defender spot check period  T", fontsize=20)
    ax1.set_ylabel(r"$J(T,\lambda^*)$", color="blue")
    ax1.tick_params(axis='y', labelcolor="blue", width=2, length=8)

    # T* representation
    ax1.axvline(T_star, color="blue", linestyle="--", linewidth=3)
    ax1.text(T_star + 0.3, np.mean(J_def), r"$T^*$",
             color="blue", fontsize=26, fontweight="bold", va="center")

    # Attacker curve
    ax3 = ax1.twinx()

    T_min, T_max = min(T_grid), max(T_grid)
    lambda_min, lambda_max = min(lambda_grid), max(lambda_grid)

    lam_to_T = lambda lam: T_min + (T_max - T_min) * (lam - lambda_min) / (lambda_max - lambda_min)
    T_for_lambda = lam_to_T(lambda_grid)

    ax3.plot(T_for_lambda, K_att, color="red")
    ax3.set_ylabel(r"$K(T^*,\lambda)$", color="red")
    ax3.tick_params(axis='y', labelcolor="red", width=2, length=8)

    # λ* representation
    x_lam = lam_to_T(lambda_star)
    ax3.axvline(x_lam, color="red", linestyle="--", linewidth=3)
    ax3.text(x_lam + 0.3, np.mean(K_att), r"$\lambda^*$",
             color="red", fontsize=26, fontweight="bold", va="center")

    # Top λ-axis
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())

    step = max(1, len(lambda_grid)//6)
    ax2.set_xticks(T_for_lambda[::step])
    ax2.set_xticklabels([f"{lam:.1f}" for lam in lambda_grid[::step]])
    ax2.set_xlabel("Attacker rate λ", fontsize=20)
    ax2.tick_params(axis='x', width=2, length=8)


    # Save figure
    # fig.savefig(
    #     f"{csv_folder}/{graph_name}_SE_plot.png",
    #     bbox_inches="tight",
    #     dpi=500,                   
    #     transparent=False,
    #     pad_inches=0.05
    # )
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------
# GENERATE THE THREE FIGURES
# ---------------------------------------------------------

for graph_name in graphs:
    plot_single_graph(graph_name)
