"""
This module provides the UNGUARD attack graph
in the same format as the MARA graph module.
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_WEIGHT_VALUE = 0  

def unguard_set_default_weight(value):
    """Set the default weight value for the entire module."""
    global DEFAULT_WEIGHT_VALUE
    DEFAULT_WEIGHT_VALUE = value


def create_UNGUARD_attack_graph(DEFAULT_WEIGHT_VALUE=DEFAULT_WEIGHT_VALUE):
    """
    Create a directed graph based on the UNGUARD structure.

    Returns:
        - nx.DiGraph: The created attack graph
        - List[int]: Nodes in topological order
    """

    graph = nx.DiGraph()
    nodes = list(range(1, 34))
    graph.add_nodes_from(nodes)
    edges = [

        # Attack Path 1: Redis
        (1, 6),
        (6, 11), (6, 12),
        (11, 16), (12, 16),
        (11, 17), (12, 17),
        (16, 23),
        (17, 24),
        (23, 29),
        (24, 30),
        (23, 10), (24, 10),
        (10, 5), (10, 7), (10, 8), (10, 9),

        # Attack Path 2: SQLi
        (1, 2), (1, 3), (1, 4),
        (2, 13), (3, 13), (4, 13),
        (13, 18),
        (18, 5),
        (5, 25),
        (25, 30),
        (25, 31),

        # Attack Path 3: Envoy Bypass
        (3, 14), (4, 14),
        (14, 19),
        (19, 26),
        (26, 7), (26, 8), (26, 9), (26, 10),
        (7, 32), (8, 32), (9, 32), (10, 32),

        # Attack Path 4: SSRF
        (2, 15), (3, 15),
        (15, 20), (15, 21), (15, 22),
        (20, 27),
        (21, 5), (21, 6),
        (22, 28),
        (27, 33), (28, 33),
        (5, 33), (6, 33),

        # Converging impacts
        (29, 33),
        (30, 33),
        (31, 33),
        (32, 33),
    ]

    graph.add_edges_from(edges)

    # -------------------------
    # Set default weights
    # -------------------------
    nx.set_edge_attributes(graph, DEFAULT_WEIGHT_VALUE, 'weight')

    # -------------------------
    # Compute topological order
    # -------------------------
    node_order = list(nx.topological_sort(graph))

    return graph, node_order


def plot_main_graph(attack_graph):
    """
    Plot the UNGUARD attack graph using a spring layout.
    """

    pos = nx.spring_layout(attack_graph, k=1.2, iterations=80, seed=42)

    plt.figure(figsize=(14, 10))

    nx.draw_networkx_edges(
        attack_graph, pos,
        edge_color='gray',
        arrows=True,
        arrowsize=20,
        width=2
    )

    nx.draw_networkx_nodes(
        attack_graph, pos,
        node_color='lightblue',
        node_size=600,
        edgecolors='darkblue',
        linewidths=1.5
    )

    nx.draw_networkx_labels(
        attack_graph, pos,
        font_size=10,
        font_weight='bold'
    )

    plt.title("UNGUARD Stackelberg Attack Graph", fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
