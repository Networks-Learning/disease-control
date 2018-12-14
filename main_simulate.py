import pandas as pd
import numpy as np 
from pprint import pprint
import matplotlib.pyplot as plt
import scipy.stats
from tqdm import tqdm
import scipy.optimize
import joblib
import networkx
import os

# from dynamics import SISDynamicalSystem
from analysis import Evaluation
from stochastic_processes import StochasticProcess, CountingProcess
from experiment import Experiment


plt.switch_backend('agg')


if __name__ == '__main__':

    '''Network'''
    df = pd.read_csv("data/contiguous-usa.txt", sep=" ", header=None)

    # construct A
    df_0, df_1 = df[0].values, df[1].values
    N, M = max(np.max(df_0), np.max(df_1)), len(df_0)
    A = np.zeros((N, N))
    for i in range(M):
        A[df_0[i] - 1, df_1[i] - 1] = 1
        A[df_1[i] - 1, df_0[i] - 1] = 1
    print('Network: ' + str(N) + ' nodes, ' + str(M) + ' edges')


    '''Initial infections'''
    infected = 10
    X_init = np.hstack(((np.ones(infected), np.zeros(N - infected))))
    X_init = np.random.permutation(X_init)


    '''Experiments'''
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


    '''Simulation (Nothing below should be changed)'''
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

