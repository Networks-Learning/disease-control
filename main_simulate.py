import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import joblib
import os

from experiment import Experiment

plt.switch_backend('agg')


if __name__ == '__main__':

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
        Experiment('test_all_but_MCM',
                   sim_dict={'total_time': 10.00, 'trials_per_setting': 30},
                   policy_dict={
                       'SOC': True,
                       'TR': True,
                       'TR-FL': True,
                       'MN': True,
                       'MN-FL': True,
                       'LN': True,
                       'LN-FL': True,
                       'LRSR': True,
                       'MCM': False,
                   }),
    ]

    # Simulation (Nothing below should be changed)
    for experiment in experiments:
        data = experiment.run(A, X_init)

        dir = 'temp_pickles/'
        filename = dir \
            + experiment.name \
            + '_Q_{}_{}_v'.format(int(experiment.cost_dict['Qlam']),
                                  int(experiment.cost_dict['Qx']))

        # create unique file for results, indexed by _v0, _v1, _v2, ...
        j = 0
        while os.path.exists(filename + str(j) + '.pkl'):
            j += 1
        final = filename + str(j) + '.pkl'
        joblib.dump(data, final)
