
# GameSec2026-Supplementary-when-to-cut-the-rope
Supplementary material for the GameSec 2026 paper: When to Cut the Rope: Optimal Timing of Cyber-Defense Interventions Against Adaptive Advanced Persistent Threats


## ε-Pessimistic Stackelberg Solver for CTR-LIPO Models

An optimization framework for computing **ε-pessimistic Stackelberg equilibria** in defender–attacker timing games driven by Cut-the-Rope (CTR)-based probabilistic attack models. The follower (attacker) is optimized with **LIPO+**, the leader (defender) is optimized with **Bayesian optimization**, and the repository produces Stackelberg-consistent deviation curves with full CSV export for reproducibility.

## Table of Contents

- [Overview](#overview)
- [Mathematical Model](#mathematical-model)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Usage](#usage)
- [Methodology](#methodology)
  - [Follower Optimization (LIPO+)](#follower-optimization-lipo)
  - [Leader Optimization (Bayesian Optimization)](#leader-optimization-bayesian-optimization)
- [Deviation Curves](#deviation-curves)
- [CTR Graph Models](#ctr-graph-models)
- [Output Files](#output-files)
- [Reproducibility](#reproducibility)
- [References](#references)

## Overview

This project computes the Stackelberg equilibrium (T*, λ*) of a defender–attacker timing game, in which:

- the defender (leader) chooses a review interval T,
- the attacker (follower) chooses an attack rate λ,
- the interaction outcome is determined by a CTR model evaluated on a chosen attack graph,
- the attacker reacts optimally to each defender's decision,
- the defender anticipates the attacker's reaction when choosing T.

Because the follower's best response need not be unique, the solver computes an **ε-pessimistic** approximation of the equilibrium rather than relying on an exact (and possibly ill-defined) best response.

## Mathematical Model

The follower solves:

λ*(T) = argmax_{λ in Λ} K(T, λ),      K(T, λ) = CTRval(T, λ) − ω · c_a(λ)

The leader anticipates this response and solves:

T* = argmin_T [ CTRval(T, λ*(T)) + γ · c_d(T) ]

where c_a and c_d are the attacker's and defender's cost functions, ω and γ are cost weights, and CTRval(T, λ) is the CTR model's success probability evaluated on the chosen attack graph.

## Repository Structure

```text
CTR_LIPO/
├── CTR/
│   └── CTR_eval_Poisson.py    # CTR engine: evaluate the Poisson distribution of 						  the attacker
    └── ctr_core               # The core algorithms for solving the security game  

├── CTR_and_LIPO_P/
│   ├── LIPO_Follower.py       # LIPO+ implementation of the follower's best 	 						 response
│   └── Plot_results.py        # Main entry point: equilibrium computation + 						    plotting
    └── Players_Deviation_from_SE.py    # Stackelberg equilibrium (SE) assessment 
 
    └── Multiple_gamma_&_omega.py      # Robustness evaluation

└── LIPO_main/
    ├── src
	   ├── LIPO_plus.py       # The adapted LIPO+ algorithms
	   ├── utils.py           # Stopping criteria of LIPO+ algorithm
              
└── graphs/
    ├── attack_graph_MARA.py 
    ├── attack_graph_MIR100.py
    ├── attack_graph_UNGUARD.py             

CSV_Images_results/
    ├── SE_results_<graph>.csv           # Numerical results (SE + deviation grids)
    └── create_<graph>_SE_plot.png       # Dual-axis deviation plot

```


## Requirements

- Python ≥ 3.9
- `numpy`
- `scipy`
- `matplotlib`
- `scikit-optimize`
- `networkx`
- `pandas`

A minimal `requirements.txt`:

```text
numpy
scipy
matplotlib
scikit-optimize
networkx
pandas
```


## Usage

The main entry points are:

```text
CTR_LIPO/CTR_and_LIPO_P/Players_Deviation_from_SE.py
```    

```text
CTR_LIPO/CTR_and_LIPO_P/Plot_results.py
``` 

Run it with:

```bash
python3 -m CTR_LIPO.CTR_and_LIPO_P. Players_Deviation_from_SE
```

```bash
python3 -m CTR_LIPO.CTR_and_LIPO_P.Plot_results
```

This will:

1. compute the ε-pessimistic Stackelberg equilibrium,
2. generate the defender and attacker deviation curves,
3. save the figure to `CSV_Images_results/`,
4. export all numerical results to a CSV file in the same directory.

## Methodology

### Follower Optimization (LIPO+)

The attacker's best response is computed using the LIPO+ global optimizer, implemented in:

```text
CTR_LIPO/CTR_and_LIPO_P/LIPO_Follower.py
```

LIPO+ is well suited to this setting because it provides global optimization guarantees for Lipschitz objectives without requiring gradient information, and is robust to the noisy evaluations produced by the CTR engine. Given the follower objective $K(T,\lambda)$, the best response is approximated by:

```python
lambda_star, J_eps = lipofollower(...)
```
The Lipschitz constant and the domain bounds for λ must be supplied by the user and should be consistent with the theoretical bounds derived for K(T, ·).

### Leader Optimization (Bayesian Optimization)

The defender anticipates the follower's reaction and solves the outer minimization over T using Bayesian optimization (gp_minimize from scikit-optimize), implemented in:

```python
Eps_Pessimistic_Stackelberg(...)
```

Bayesian optimization is used here because each evaluation of the leader's objective requires solving the inner follower problem, making derivative-free, sample-efficient optimization preferable to grid search or gradient-based methods.

## Deviation Curves

To visualize and validate the computed equilibrium, the script generates two deviation curves.

**1. Defender deviation curve.** For each candidate T, the attacker recomputes its best response λ*(T), and the curve plots:

J(T, λ*(T))

**2. Attacker deviation curve.** The defender commits to the equilibrium T*, and the curve plots the attacker's payoff as λ varies:

K(T*, λ)

Both curves are plotted on dual axes and saved as:

```text
Optimal_vs_suboptimal_SE_<graph>.png
```

## CTR Graph Models

The CTR engine is provided in:

```text
CTR_LIPO/CTR/CTR_eval_Poisson.py
```

You can switch between attack graph models:

- `create_mara_attack_graph`
- `create_mir100_attack_graph`
- `create_UNGUARD_attack_graph`

by modifying:

```python
graph_name = get_graph_name(create_mara_attack_graph)
```
```python
graph_considered in run_experiment function of CTR_eval_poisson

```

## Output Files

All numerical results are exported to:

```text
SE_results_<graph>.csv
```
The exported data includes:

| Content | Description |
|---|---|
| Defender deviation grid | Grid of T values used for the defender deviation curve |
| Attacker deviation grid | Grid of λ values used for the attacker deviation curve |
| Defender payoff curve | J(T, λ*(T)) evaluated on the defender grid |
| Attacker payoff curve | K(T*, λ) evaluated on the attacker grid |
| Equilibrium values | T* and λ * |


This ensures that every figure produced by the repository can be regenerated and audited from the exported CSV data alone.

## Reproducibility

All figures are saved as high-resolution PNG files and accompanied by their underlying numerical data in CSV form, so that results can be independently verified or replotted without rerunning the optimization.

## References

If you use this code, please cite:

**LIPO+:**
> LIPO+: Frugal global optimization for lipschitz functions, Serr{\'e}, Ga{\"e}tan and Beja-Battais, Perceval and Chirrane, Sophia and Kalogeratos, Argyris and Vayatis, Nicolas *Pro. of the 13th Hellenic Conference on Artificial Intelligence*, 2024.

**CTR-LIPO Stackelberg model:**
> `Arnold Willie Kouam Kounchou, Stefan Rass`, "When to Cut the Rope: Optimal Timing of Cyber-Defense Interventions Against Adaptive Advanced Persistent Threats," *International Conference on Game Theory and AI for Security*, `2026`.


