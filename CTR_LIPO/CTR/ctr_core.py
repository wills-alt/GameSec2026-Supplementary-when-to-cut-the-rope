"""
Core functionality for analyzing and solving security games on attack graphs.

This module provides the core algorithms for:
1. Processing attack graphs
2. Calculating payoff distributions
3. Solving security games using linear programming
"""

import networkx as nx
import numpy as np
from scipy.optimize import linprog

import logging


# Configure logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


DEFAULT_WEIGHT_VALUE = 0  # Default fallback value
DEBUG_MODE = False  # Global debug flag

def core_set_default_weight(value):
    """Set the default weight value for the entire module."""
    global DEFAULT_WEIGHT_VALUE
    DEFAULT_WEIGHT_VALUE = value

def core_set_debug_mode(enabled=False):
    """Toggle debug output on or off."""
    global DEBUG_MODE
    DEBUG_MODE = enabled


# ------------------ PREPROCESSING ------------------

def find_and_add_entry_node(graph):
    """
    Add a virtual entry node to a graph connecting to all root nodes.
    Args:  
        graph: NetworkX graph to process 
    Returns:
        Tuple of (entry_node, modified_graph, original_roots)
    """
    # First identify the original root nodes
    original_roots = [n for n, deg in graph.in_degree() if deg == 0]
    
    if len(original_roots) > 1:
        # add virtual entry node
        entry = 0  
        graph.add_node(entry)
        for r in original_roots:
            graph.add_edge(entry, r, weight=DEFAULT_WEIGHT_VALUE)
        return entry, graph, original_roots
    else:
        # Only one root, use it as entry
        entry = original_roots[0]
        return entry, graph, original_roots

def merge_targets_with_multi_edges(orig_graph):
    """
    Merges all target nodes into a single virtual target node.
    Preserves parallel edges and their weights from the original graph.
    Args:
        orig_graph: Original NetworkX graph
    Returns:
        Modified graph with merged target nodes
    """
    # Find target nodes (nodes with no outgoing edges)
    targets = [n for n, out_degree in orig_graph.out_degree() if out_degree == 0]
    
    # Return original graph if 0 or 1 target
    if len(targets) <= 1:
        return orig_graph

    # Create merged label for virtual target
    merged_label = "c(" + ",".join(str(t) for t in targets) + ")"

    # Create new MultiDiGraph
    newG = nx.MultiDiGraph()

    # Add all non-target nodes
    non_targets = [n for n in orig_graph.nodes() if n not in targets]
    for node in non_targets:
        newG.add_node(node)
        
    # Add the virtual target node
    newG.add_node(merged_label)
    
    # Track edges to target nodes
    pred_target_edges = {}

    # Collect all edges going to target nodes
    for u, v, data in orig_graph.edges(data=True):
        if v in targets:
            if u not in pred_target_edges:
                pred_target_edges[u] = []
            weight = data.get('weight', DEFAULT_WEIGHT_VALUE)
            pred_target_edges[u].append((weight, v))

    # Create edges to virtual target preserving parallel edges
    for u, edges in pred_target_edges.items():
        weight_counts = {}
        for weight, _ in edges:
            weight_counts[weight] = weight_counts.get(weight, 0) + 1

        for weight, count in weight_counts.items():
            for _ in range(count):
                newG.add_edge(u, merged_label, weight=weight)
    
    # Copy over all other edges
    for u, v, data in orig_graph.edges(data=True):
        if v not in targets and u not in targets:
            newG.add_edge(u, v, **data)

    return newG

def generate_game_elements(graph, entry_node, original_roots):
    """
    Set up all elements needed for security game after preprocessing the graph.
    Args:
        graph: Preprocessed attack graph
        entry_node: The virtual entry node
        original_roots: List of original root nodes
        
    Returns:
        Tuple containing:
        (routes, V, as1, as2, target_list, node_order, adv_list, theta, m)
    """
    # Find target node
    target_list = [n for n, d in graph.out_degree() if d == 0]
    
    # Get all possible attack routes
    raw_routes = list(nx.all_simple_paths(graph, entry_node, target_list[0]))

    # Remove duplicate paths
    consolidated_routes = []
    seen_paths = set()

    for path in raw_routes:
        path_key = tuple(path)
        if path_key not in seen_paths:
            seen_paths.add(path_key)
            consolidated_routes.append(list(path))

    routes = consolidated_routes

    # Get all unique nodes appearing in any route
    V = sorted(set(node for path in routes for node in path), key=str)

    # Get nodes in topological order
    topo_all = list(nx.topological_sort(graph))
    node_order = [n for n in topo_all if n in V]
    
    # Create list of defender check locations (excluding entry, target, roots)
    excluded = {entry_node} | set(target_list) | set(original_roots)
    as1 = [n for n in V if n not in excluded]

    # Set up attack paths
    as2 = routes

    # Create list of possible attacker locations (excluding entry and target)
    excluded_nodes = set([entry_node]) | set(target_list)
    adv_list = [n for n in V if n not in excluded_nodes]
    
    if len(adv_list) == 0:
        logger.warning("No adversary intermediate locations found. Check graph structure.")

    # Calculate initial probabilities for attacker locations
    theta = {loc: 1/len(adv_list) for loc in adv_list} if adv_list else {}
    
    # Count total number of attack paths
    m = len(routes)
    
    return routes, V, as1, as2, target_list, node_order, adv_list, theta, m

def lossDistribution(U):
    """
    Creates standardized format for probability distribution.
    Args:
        U: Array of normalized probabilities
        
    Returns:
        Dictionary with distribution attributes
    """
    return {
        'dpdf': U,  
        'support': np.arange(1, len(U) + 1), # values of the distribution
        'cdf': np.cumsum(U), ## 𝑃(loss <= 𝑘)
        'tail': 1 - np.cumsum(U) + U,  ## 𝑃(loss≥𝑘)
        'range': [1, len(U)]
    }

def calculate_payoff_distribution(graph, as1, as2, V, adv_list, theta, random_steps_fn, 
                                 attack_rate, defense_rate, node_order):
    """
    Calculate probability distributions for each check location & attack path pair.
    
    Args:
        graph: Attack graph
        as1: List of defender check locations
        as2: List of attack paths
        V: List of all nodes in any path
        adv_list: List of possible attacker positions
        theta: Dictionary of starting position probabilities
        random_steps_fn: Function to calculate random walk probabilities
        attack_rate: Attacker movement rate parameter
        defense_rate: Defender check rate parameter
        node_order: Topological ordering of nodes
        
    Returns:
        List of probability distributions for each check+path pair
    """
    payoffs = []

    for check in as1:
        for path in as2:
            U = np.zeros(len(V))

            for avatar in adv_list:
                L = np.zeros(len(V))

                if avatar in path:
                    # Extract relevant portion of path from avatar position
                    start_idx = path.index(avatar)
                    route = path[start_idx:]
                    
                    # Get raw movement probabilities
                    pdf_d = random_steps_fn(route, attack_rate, defense_rate, graph)

                    # Adjust based on defender's check point
                    if check in route:
                        check_idx = route.index(check)
                        # Add 1 to include the check point
                        cutPoint = check_idx + 1
                    else:
                        cutPoint = len(route)

                    # Take probabilities up to check point and renormalize
                    pdf_subset = pdf_d[:cutPoint]
                    if np.sum(pdf_subset) < 1e-15:
                        payoffDistr = np.zeros(cutPoint)
                        payoffDistr[-1] = 1.0
                    else:
                        payoffDistr = pdf_subset / np.sum(pdf_subset)
                    
                    # Map probabilities to node indices in V
                    route_subset = route[:cutPoint]
                    for idx_node, node in enumerate(route_subset):
                        L[V.index(node)] = payoffDistr[idx_node]

                else:
                    # If avatar not on path, it stays at current position
                    L[V.index(avatar)] = 1.0

                # Weight by probability of starting at this position
                U += theta[avatar] * L

            # Normalize and handle edge cases
            U_sum = np.sum(U)
            if U_sum < 1e-15:
                U = np.full_like(U, 1e-7)
            else:
                # normalize and prevent 0 probabilities
                U = U/U_sum
                U = np.where(U < 1e-7, 1e-7, U)
            
            # Reorder according to topological sort
            node_positions = [V.index(n) for n in node_order]
            U = U[node_positions]

            # Create final distribution
            ld = lossDistribution(U)
            payoffs.append(ld)

    return payoffs

def solve_game(payoffs, as1, as2):
    """
    Solve zero-sum security game using linear programming.
    Args:
        payoffs: List of payoff distributions
        as1: List of defender check locations
        as2: List of attack paths
        
    Returns:
        Dictionary containing equilibrium strategies and values
    """
    n = len(as1)
    m = len(as2)
    
    # Create payoff matrix
    payoff_matrix = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            idx = i*m + j
            ld = payoffs[idx]
            payoff_matrix[i, j] = ld['dpdf'][-1]
    
    ### Defender's LP optimization
    c = np.zeros(n+1)  
    c[0] = 1.0
    
    A_ub = np.zeros((m, n+1))
    b_ub = np.zeros(m)
    for j in range(m):
        A_ub[j,0] = -1.0
        for i in range(n):
            A_ub[j,i+1] = payoff_matrix[i,j]
            
    A_eq = np.zeros((1, n+1))
    A_eq[0,1:] = 1.0
    b_eq = np.array([1.0])
    
    bounds = [(0,None)]*(n+1)
    
    v_defender = None
    v_attacker = None
    
    # Solve LP for defender
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    
    if res.success:
        v_defender = res.x[0]
        x_def = res.x[1:]
        
        ### Attacker's optimization
        c_att = np.zeros(m+1)
        c_att[0] = -1.0
        
        A_ub_att = np.zeros((n, m+1))
        b_ub_att = np.zeros(n)
        for i in range(n):
            A_ub_att[i,0] = 1.0
            for j in range(m):
                A_ub_att[i,j+1] = -payoff_matrix[i,j]
                
        A_eq_att = np.zeros((1, m+1))
        A_eq_att[0,1:] = 1.0
        b_eq_att = np.array([1.0])
        
        bounds_att = [(0,None)]*(m+1)

        # Solve LP for attacker
        res_att = linprog(c_att, A_ub=A_ub_att, b_ub=b_ub_att, 
                         A_eq=A_eq_att, b_eq=b_eq_att, bounds=bounds_att)
        
        if res_att.success:
            y_att = res_att.x[1:]
            v_attacker = res_att.x[0]
            
            # Check if values match
            if abs(v_defender - v_attacker) > 1e-5:
                logger.warning("\nWarning: Defender and attacker values don't match!")
                logger.warning(f"Defender value: {v_defender:.6f}")
                logger.warning(f"Attacker value: {v_attacker:.6f}")
            
            return {
                'optimal_defense': dict(zip(as1, x_def)),
                'attacker_strategy': y_att,
                'defender_success': v_defender,
                'attacker_success': v_attacker
            }
    
    logger.warning("LP optimization failed")
    return None



def run_game(attacker_graph, defender_graph=None, attack_rate_list=None, dropped=None, 
             defense_rate_list=None, random_steps_fn=None):
    """
    Run security game analysis on attack graphs.
    
    Args:
        attacker_graph: Graph from attacker's perspective
        defender_graph: Graph from defender's perspective (default: same as attacker)
        attack_rate_list: List of attacker movement rates to analyze
        dropped: List of nodes dropped from defender's view
        defense_rate_list: List of defender check rates to analyze
        random_steps_fn: Function to calculate random walk probabilities
        
    Returns:
        Final equilibrium results
    """
    # For backward compatibility and initial testing
    if defender_graph is None:
        defender_graph = attacker_graph

    # Process attacker graph
    atk_virtual_entry_node, attacker_graph, atk_original_roots = find_and_add_entry_node(attacker_graph)
    attacker_graph = merge_targets_with_multi_edges(attacker_graph) 
    # Calculate attacker elements
    _, V, _, as2, target_list, node_order, adv_list, theta, m = generate_game_elements(
        attacker_graph, atk_virtual_entry_node, atk_original_roots)

    # Process defender graph
    def_virtual_entry_node, defender_graph, def_original_roots = find_and_add_entry_node(defender_graph)
    defender_graph = merge_targets_with_multi_edges(defender_graph) 
    # Calculate defender elements
    _, _, as1, _, _, _, _, _, _ = generate_game_elements(defender_graph, def_virtual_entry_node, def_original_roots)
    
    # Set default rate lists if not provided
    if not defense_rate_list:
        defense_rate_list = [0]
    if not attack_rate_list:
        attack_rate_list = [0]
    


    results = {}   # {(defRate, attRate): eq or None}
    final_eq = None

    for defenseRate in defense_rate_list:
        for attackRate in attack_rate_list:
            payoffs = calculate_payoff_distribution(
                attacker_graph, as1, as2, V, adv_list, theta,
                random_steps_fn,
                attackRate, defenseRate, node_order
            )
            eq = solve_game(payoffs, as1, as2)
            results[(defenseRate, attackRate)] = eq
            if eq is not None:
                final_eq = eq

    # Return both the last successful eq (backwards compatible) and the full results
    return {'last_equilibrium': final_eq, 'all_equilibria': results}




def main(full_attack_graph, attack_rate_list=None, defense_rate_list=None, random_steps_fn=None):
    """
    Main entry point for running security game analysis.
    
    Args:
        full_attack_graph: Complete attack graph
        defender_subgraphs_list: List of (subgraph, dropped_nodes) tuples
        attack_rate_list: List of attacker movement rates to analyze
        defense_rate_list: List of defender check rates to analyze
        random_steps_fn: Function to calculate random walk probabilities
        
    Returns:
        Baseline results and optional subgraph analysis results
    """

    baseline_result = run_game(
        attacker_graph=full_attack_graph, 
        defender_graph=full_attack_graph, 
        attack_rate_list=attack_rate_list, 
        defense_rate_list=defense_rate_list, 
        random_steps_fn=random_steps_fn
    )
    
    return baseline_result

