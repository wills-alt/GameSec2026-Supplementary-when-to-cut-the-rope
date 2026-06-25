"""
CTR optimal value evaluation: Basic Attack Graph Analysis

This script analyzes security games on attack graphs with for complete
defender visibility.
"""

import os
import logging
import numpy as np
import pathlib
import argparse
from datetime import datetime
from scipy.stats import poisson

# Import ctr library components
from ..graphs.attack_graph_MARA import create_mara_attack_graph
from ..graphs.attack_graph_MARA import mara_set_default_weight
from ..graphs.attack_graph_MARA import plot_main_graph

from ..graphs.attack_graph_MIR100 import *


from .ctr_core import main 
from .ctr_core import core_set_default_weight
from .ctr_core import core_set_debug_mode


# # Configure logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# Default values - all defaults are defined here and used consistently throughout the code
DEFAULT_WEIGHT_VALUE = 0
DEFAULT_ATTACK_RATES = [2, 3]
DEFAULT_DEFENSE_RATES = [1]

######## Define random steps function for attacker movement ###################


def random_steps(route, attack_rate=None, defense_rate=None, graph=None):
    """
    Calculate probability distribution for random steps using Poisson distribution.
    Args:
        route: Attack path
        attack_rate: Lambda parameter for Poisson distribution
        defense_rate: Not used in this implementation
        graph: Attack graph
        
    Returns:
        Array of probabilities for each position in the path
    """
    length = len(route)
    if attack_rate is None:
        attack_rate = DEFAULT_ATTACK_RATES[0]  # Use default attack rate
    else:
        attack_rate = float(attack_rate)      # attacker aggressivity

    if defense_rate is None:
        defense_rate = DEFAULT_DEFENSE_RATES[0]  # Use default attack rate
    else:
        defense_rate  = float(defense_rate)     # defender period

    lambda_round = attack_rate * (defense_rate) 

    # Get PMF for values 0 to length-1
    pmf = poisson.pmf(np.arange(length), lambda_round) 

    # Normalize (though poisson.pmf should already sum to ~1)
    pmf = pmf / pmf.sum()
    return pmf


################################ run_experiment ################################
################################################################################

def run_experiment(attack_rate=None, defense_rate=None, graph_considered=create_mara_attack_graph):
    """Run the core analysis with configured parameters."""

    # Set defaults
    mara_set_default_weight(DEFAULT_WEIGHT_VALUE)
    core_set_default_weight(DEFAULT_WEIGHT_VALUE)
    core_set_debug_mode(False)
    if attack_rate == None: 
        attack_rate = DEFAULT_ATTACK_RATES
    if defense_rate == None:
        defense_rate = DEFAULT_DEFENSE_RATES
    # Create attack graph
    full_attack_graph, node_order = graph_considered(DEFAULT_WEIGHT_VALUE)

    baseline_result = main(
        full_attack_graph = full_attack_graph,
        attack_rate_list = attack_rate,
        defense_rate_list = defense_rate,
        random_steps_fn = random_steps
    )
    
    return baseline_result


if __name__ == "__main__":
    
    # Pass logger to run_experiment to avoid duplicate setup
    Opt_success_prob = run_experiment([2, 3], [1])
    all_eq = Opt_success_prob['all_equilibria']

    for (def_rate, att_rate), eq in sorted(all_eq.items()):
        d = eq.get('defender_success')
        a = eq.get('attacker_success')
        logger.info(f"Rates (def={def_rate}, att={att_rate}): defender_success={d:.8f}, attacker_success={a:.8f}")

