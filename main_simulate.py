"""

Main script to simulate experiments

"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import joblib
import os

from experiment import Experiment

OUTPUT_DIR = 'temp_pickles/'


def build_filename(exp):
    """
    Find and construct an available filename for the experiment `exp`.
    The file is index by _v0, _v1, _v2, ...
    """
    filename_prefix = (f"{exp.name:s}_"
                       f"Q_{exp.cost_dict['Qlam']:.0f}_"
                       f"{exp.cost_dict['Qx']:.0f}")
    filepath_prefix = os.path.join(OUTPUT_DIR, filename_prefix)
    j = 0
    filename = f"{filepath_prefix}_v{j}.pkl"
    while os.path.exists(filename):
        j += 1
        filename = f"{filepath_prefix}_v{j}.pkl"
    return filename


if __name__ == '__main__':

    plt.switch_backend('agg')

    # Construct the adjacency  matrix A of the propagation network
    net = nx.read_edgelist('data/contiguous-usa.txt')
    A = nx.adjacency_matrix(net).toarray().astype(float)
    n_nodes = net.number_of_nodes()
    n_edges = net.number_of_edges()
    print(f"Network: {n_nodes:d} nodes, {n_edges:d} edges")

    # Initial infections
    infected = 10
    X_init = np.hstack(((np.ones(infected), np.zeros(n_nodes - infected))))
    X_init = np.random.permutation(X_init)

    # Experiments
    experiments = [
        Experiment('test_all',
                   sim_dict={
                      'total_time': 10.00,
                      'trials_per_setting': 30
                   },
                   param_dict={
                       'beta':  6.0,
                       'gamma': 5.0,
                       'delta': 1.0,
                       'rho':   5.0,
                       'eta':   1.0
                   },
                   cost_dict={
                     'Qlam': 1.0,
                     'Qx': 300.0
                   },
                   policy_list=[
                       'SOC',
                       'TR', 'TR-FL',
                       'MN', 'MN-FL',
                       'LN', 'LN-FL',
                       'LRSR',
                       'MCM',
                   ],
                   baselines_dict={
                       'TR': 0.003,
                       'MN': 0.0007,
                       'LN': 0.0008,
                       'LRSR': 22.807,
                       'MCM': 22.807,
                       'FL_info': dict(N=356.1000, max_u=19.5352),
                   })
    ]

    # Simulation (Nothing below should be changed)
    for exp in experiments:
        print(f"Running experiment `{exp.name}`...")
        data = exp.run(A, X_init)

        filename = build_filename(exp)
        print(f"Save the simulation to: {filename:s}")
        joblib.dump(data, filename)
