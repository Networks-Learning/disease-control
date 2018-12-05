"""
Implements the MaxCutMinimization (MCM) strategy to suppress SIS epidemics, as
defined in the following paper.

K. Scaman, A. Kalogeratos and N. Vayatis, "Suppressing Epidemics in Networks
Using Priority Planning," in IEEE Transactions on Network Science and
Engineering, vol. 3, no. 4, pp. 271-285, Oct.-Dec. 2016.
"""
import networkx as nx
import numpy as np
import scipy as sp
import math


def _spectral_sequencing(adjacency_mat):
    """
    Order the nodes according to the eigenvector related to the second
    smallest eigenvalue of the Laplacian matrix.
    """
    degree_mat = np.diag(adjacency_mat.sum(axis=0))
    laplacian_mat = degree_mat - adjacency_mat
    # Compute the second largest eigenvalue
    _, e_2 = sp.linalg.eigh(laplacian_mat, eigvals=(1, 1))
    return np.argsort(e_2.squeeze())


def _swap_heuristic(adjacency_mat, plist, seed, n_swaps):
    """
    Randomized heuristic that starts from the priority list `plist` and tries
    to improve the maxcut.  At each iteration, it applies one random swap  and
    keep the modifications if this swap results in a lower sum of cuts.
    """
    G = nx.Graph(adjacency_mat)
    n_nodes = len(adjacency_mat)
    # Initialize the random seed
    if seed:
        np.random.seed(seed)
    # Initialize the current sum of cuts
    curr_sum_cuts = sum(cut_list(G, plist))
    for i in range(n_swaps):
        # Sample two nodes ramdomly
        x, y = np.random.randint(0, n_nodes, size=2)
        # Swap their order
        plist[x], plist[y] = plist[y], plist[x]
        # Compute the new sum of cuts
        new_sum_cuts = sum(cut_list(G, plist))
        # If improvement, update the sum of cuts, else ignore
        if new_sum_cuts < curr_sum_cuts:
            curr_sum_cuts = new_sum_cuts
        else:
            plist[x], plist[y] = plist[y], plist[x]
    return plist


def mcm(adjacency_mat, seed=None, n_swaps=None):
    """
    Compute the MaxCutMinimization priority planning based on spectral
    sequencing and a random swap heuristic.
    """
    if not isinstance(adjacency_mat, np.ndarray):
        raise TypeError('The adjacency matrix must be a numpy ndarray.')
    if not len(adjacency_mat.shape) == 2:
        raise ValueError('The adjacency matrix should be of dimension 2.')
    if adjacency_mat.shape[0] != adjacency_mat.shape[1]:
        raise ValueError('The adjacency matrix should squared.')
    if not np.allclose(adjacency_mat, adjacency_mat.T):
        raise ValueError('The adjacency matrix should be symmetric.')
    plist = _spectral_sequencing(adjacency_mat)
    if n_swaps is None:
        n_swaps = len(adjacency_mat)
    plist = _swap_heuristic(adjacency_mat, plist, seed, n_swaps)
    return plist


def cut_list(adjacency_mat, plist):
    """
    Compute the cuts of the priority list 'plist'. Return an array where the
    element in position `i` is the cut between nodes `i` and `i+1` in the
    priority list.
    """
    n_nodes = len(adjacency_mat)
    G = nx.Graph(adjacency_mat)
    cut_list = np.zeros(n_nodes, dtype='int')
    visited = set()
    pending_edges = 0
    for x, i in zip(plist, range(n_nodes)):
        for neigh in G.neighbors(x):
            if neigh not in visited:
                pending_edges += 1
            else:
                pending_edges -= 1
            cut_list[i] = pending_edges
        visited.add(x)
    return cut_list
