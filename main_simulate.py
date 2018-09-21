import pandas as pd
import numpy as np 
from pprint import pprint
import matplotlib.pyplot as plt
import scipy.stats
from tqdm import tqdm
import scipy.optimize
import joblib
import networkx

from dynamics import SISDynamicalSystem
from analysis import Evaluation
from stochastic_processes import StochasticProcess, CountingProcess

plt.switch_backend('agg')

if __name__ == '__main__':

    from_file = True
    if from_file:
        # df = pd.read_csv("data/facebook-60k-new-orleans-MPI.txt", sep="\t", header=None)
        # df = pd.read_csv("data/facebook-ego-4k-NIPS12.txt", sep=" ", header=None)
        df = pd.read_csv("data/contiguous-usa.txt", sep=" ", header=None)

        '''Construct network from input'''
        df_0, df_1 = df[0].values, df[1].values
        N, M = max(np.max(df_0), np.max(df_1)), len(df_0)
        A = np.zeros((N, N))
        for i in range(M):
            A[df_0[i] - 1, df_1[i] - 1] = 1
            A[df_1[i] - 1, df_0[i] - 1] = 1
        print('Network: ' + str(N) + ' nodes, ' + str(M) + ' edges')
    
    else:
        '''Undirected k-regular graph with N nodes'''
        N = 50
        k = 5
        G = networkx.random_regular_graph(k, N, seed=0).to_undirected()
        A = networkx.to_numpy_array(G)
        print(networkx.info(G))

    '''Initial infections'''
    infected = 10
    X_init = np.hstack(((np.ones(infected), np.zeros(N - infected))))
    # X_init = np.random.permutation(X_init)


    '''Simulation'''
    trials_per_setting = 30
    name = 'etas'
    settings = [ {
        'param': {'beta':  6.0,
                  'gamma': 5.0,
                  'delta': 1.0,
                  'rho':   5.0,
                  'eta':   0.01},

        'cost': {'Qlam': 1.0 * np.ones(N),
                 'Qx': 10.0 * np.ones(N)},

        'time': {'total': 10.00,
                 'dt': 0.0001}

    }, {
        'param': {'beta':  6.0,
                  'gamma': 5.0,
                  'delta': 1.0,
                  'rho':   5.0,
                  'eta':   0.001},

        'cost': {'Qlam': 1.0 * np.ones(N),
                 'Qx': 10.0 * np.ones(N)},

        'time': {'total': 10.00,
                 'dt': 0.0001}

    }]
    for ind, setting in enumerate(settings):

        '''Model parameters'''
        param =  setting['param']
        cost = setting['cost']
        time = setting['time']

        '''Simulate trajectory of dynamical system under various heuristics'''
        system = SISDynamicalSystem(N, X_init, A, param, cost)

        
        data_opt, data_MN_degree_heuristic, data_trivial = [], [], []
        for tr in range(trials_per_setting):
            print("Trial # " + str(tr))

            # stochastic optimal control
            data_opt.append(system.simulate_opt(time, plot=False))
            
            # # MN degree heuristic
            data_MN_degree_heuristic.append(system.simulate_MN_degree_heuristic(1.0, time, plot=False))

            # # Trivial heuristic
            data_trivial.append(system.simulate_trivial(1.0, time, plot=False))

        '''Store data'''
        data = [data_opt, data_MN_degree_heuristic, data_trivial]
        joblib.dump(data, 'temp_pickles/results_' + name 
                                                  + '_' 
                                                  + str(ind) 
                                                  + '__US_' 
                                                  + str(trials_per_setting) 
                                                  + '_opt_deg_5triv.pkl')

