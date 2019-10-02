import itertools
import json

import pandas as pd
import numpy as np
import networkx as nx

from .settings import EBOLA_BASE_GRAPH_FILE, EBOLA_SCALED_GRAPH_FILE


def make_ebola_network(n_nodes, p_in, p_out, seed=None):
    """
    Build the EBOLA network with `n_nodes` based on the network of connected
    districts. Each district is mapped into a cluster of size proportional to
    the population of the district.

    Arguments:
    ==========
    n_nodes : int
        Desired number of nodes. Note: the resulting graph may have one node
        more or less than this number due to clique size approximation.

    Return:
    =======
    graph : networkx.Graph
        Undirected propagation network
    """
    # Load base graph
    with open(EBOLA_BASE_GRAPH_FILE, 'r') as f:
        base_graph_data = json.load(f)
    base_graph = nx.readwrite.json_graph.node_link_graph(base_graph_data)
    cluster_names = list(base_graph.nodes())
    country_names = [d['country'] for n, d in base_graph.nodes(data=True)]
    cluster_sizes = [int(np.ceil(n_nodes * base_graph.node[u]['size']))
                     for u in cluster_names]
    nodes_district_name = np.repeat(cluster_names, cluster_sizes)
    nodes_country_name = np.repeat(country_names, cluster_sizes)
    n_nodes = sum(cluster_sizes)
    # Build the intra/inter cluster probability matrix
    base_adj = nx.adjacency_matrix(base_graph).toarray().astype(float)
    base_adj[base_adj == 1] = p_out
    base_adj[np.eye(len(base_graph.nodes()), dtype=bool)] = p_in
    # Generate stoch block model graph
    graph = nx.generators.stochastic_block_model(cluster_sizes, base_adj, seed=seed)
    # Assign district attribute to each node
    for u, district, country in zip(graph.nodes(), nodes_district_name, nodes_country_name):
        graph.node[u]['district'] = district
        graph.node[u]['country'] = country
    # Sanity check for name assignment of each cluster
    num_unique_block_district = len(set([(node_data['block'], node_data['district']) for u, node_data in graph.nodes(data=True)]))
    assert num_unique_block_district == len(cluster_names)
    # Extract the giant component
    graph = max(nx.connected_component_subgraphs(graph), key=len)
    return graph
