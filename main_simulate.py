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
    X_init = np.random.permutation(X_init)


    '''Simulation'''
    trials_per_setting = 50
    settings = [
        {'param': {
            'beta':  6.0,
            'gamma': 5.0,
            'delta': 1.0,
            'rho':   5.0,
            'eta':   1.0},
        'cost': {
            'Qlam': 1.0 * np.ones(N),
            'Qx': 1.0 * np.ones(N)},
        'time': {
            'total': 10.00
        },
            'baselines': {
            'trivial': 0.35,
            'MN_deg': 0.09,
            'LN_deg': 0.10,  
            'LRSR': 5.0, 
            'CURE': 1.0,  # TODO
            'MCM': 1.0,  # TODO
            'frontld': dict(N=163.3000, peak_OPT=18.7468)
        }
        },
        {'param': {
            'beta':  6.0,
            'gamma': 5.0,
            'delta': 1.0,
            'rho':   5.0,
            'eta':   1.0},
        'cost': {
            'Qlam': 1.0 * np.ones(N),
            'Qx': 10.0 * np.ones(N)},
        'time': {
            'total': 10.00
        },
            'baselines': {
            'trivial': 0.25,
            'MN_deg': 0.06,
            'LN_deg': 0.10, 
            'LRSR': 50.0, 
            'CURE': 1.0,  # TODO
            'MCM': 1.0,  # TODO
            'frontld': dict(N=719.4000, peak_OPT=100.0833)
        }
        },
        {'param': {
            'beta':  6.0,
            'gamma': 5.0,
            'delta': 1.0,
            'rho':   5.0,
            'eta':   1.0},
        'cost': {
            'Qlam': 1.0 * np.ones(N),
            'Qx': 20.0 * np.ones(N)},
        'time': {
            'total': 10.00
        },
            'baselines': {
            'trivial': 0.2,
            'MN_deg': 0.045,
            'LN_deg': 0.06, 
            'LRSR': 90.0, 
            'CURE': 1.0,  # TODO
            'MCM': 1.0,  # TODO
            'frontld': dict(N=881.1667, peak_OPT=138.0000)
        }
        },
        {'param': {
            'beta':  6.0,
            'gamma': 5.0,
            'delta': 1.0,
            'rho':   5.0,
            'eta':   1.0},
        'cost': {
            'Qlam': 1.0 * np.ones(N),
            'Qx': 50.0 * np.ones(N)},
        'time': {
            'total': 10.00
        },
            'baselines': {
            'trivial': 0.1,
            'MN_deg': 0.03,
            'LN_deg': 0.04, 
            'LRSR': 100.0, 
            'CURE': 1.0,  # TODO
            'MCM': 1.0,  # TODO
            'frontld': dict(N=991.1000, peak_OPT=192.4839)
        }
        },
        {'param': {
            'beta':  6.0,
            'gamma': 5.0,
            'delta': 1.0,
            'rho':   5.0,
            'eta':   1.0},
        'cost': {
            'Qlam': 1.0 * np.ones(N),
            'Qx': 100.0 * np.ones(N)},
        'time': {
            'total': 10.00
        },
            'baselines': {
            'trivial': 0.035,
            'MN_deg': 0.015,
            'LN_deg': 0.015, 
            'LRSR': 90.0,
            'CURE': 1.0,   # TODO
            'MCM': 1.0,  # TODO
            'frontld': dict(N=911.0333, peak_OPT=229.0777)
        }
        },
        {'param': {
            'beta':  6.0,
            'gamma': 5.0,
            'delta': 1.0,
            'rho':   5.0,
            'eta':   1.0},
        'cost': {
            'Qlam': 1.0 * np.ones(N),
            'Qx': 150.0 * np.ones(N)},
        'time': {
            'total': 10.00
        },
            'baselines': {
            'trivial': 0.018,
            'MN_deg': 0.004,
            'LN_deg': 0.004, 
            'LRSR': 50.0, 
            'CURE': 1.0,  # TODO
            'MCM': 1.0,  # TODO
            'frontld': dict(N=732.8000, peak_OPT=239.8862)
        }
        },
        {'param': {
            'beta':  6.0,
            'gamma': 5.0,
            'delta': 1.0,
            'rho':   5.0,
            'eta':   1.0},
        'cost': {
            'Qlam': 1.0 * np.ones(N),
            'Qx': 200.0 * np.ones(N)},
        'time': {
            'total': 10.00
        },
            'baselines': {
            'trivial': 0.01,
            'MN_deg': 0.0030,
            'LN_deg': 0.0030, 
            'LRSR': 40.0, 
            'CURE': 1.0,  # TODO
            'MCM': 1.0,  # TODO
            'frontld': dict(N=615.8667, peak_OPT=241.1438)
        }
        },
        {'param': {
            'beta':  6.0,
            'gamma': 5.0,
            'delta': 1.0,
            'rho':   5.0,
            'eta':   1.0},
        'cost': {
            'Qlam': 1.0 * np.ones(N),
            'Qx': 300.0 * np.ones(N)},
        'time': {
            'total': 10.00
        },
            'baselines': {
            'trivial': 0.003,
            'MN_deg': 0.0008,
            'LN_deg': 0.0008, 
            'LRSR': 22.0,
            'CURE': 1.0,  # TODO
            'MCM': 1.0,  # TODO
            'frontld': dict(N=356.1000, peak_OPT=240.9349)
        }
        },
        {'param': {
            'beta':  6.0,
            'gamma': 5.0,
            'delta': 1.0,
            'rho':   5.0,
            'eta':   1.0},
        'cost': {
            'Qlam': 1.0 * np.ones(N),
            'Qx': 400.0 * np.ones(N)},
        'time': {
            'total': 10.00
        },
            'baselines': {
            'trivial': 0.0017,
            'MN_deg': 0.0004,
            'LN_deg': 0.0004, 
            'LRSR': 8.0, 
            'CURE': 1.0,  # TODO
            'MCM': 1.0,  # TODO
            'frontld': dict(N=280.0331, peak_OPT=253.9399)
        }
        },
        {'param': {
            'beta':  6.0,
            'gamma': 5.0,
            'delta': 1.0,
            'rho':   5.0,
            'eta':   1.0},
        'cost': {
            'Qlam': 1.0 * np.ones(N),
            'Qx': 500.0 * np.ones(N)},
        'time': {
            'total': 10.00
        },
            'baselines': {
            'trivial': 0.00077,
            'MN_deg': 0.00018,
            'LN_deg': 0.00018, 
            'LRSR': 6.0, 
            'CURE': 1.0,  # TODO
            'MCM': 1.0,  # TODO
            'frontld': dict(N=166.6667, peak_OPT=287.7502)
        }
        },

    ]


    for ind, setting in enumerate(settings):

        '''Model parameters'''
        param =  setting['param']
        cost = setting['cost']
        time = setting['time']

        '''Simulate trajectory of dynamical system under various heuristics'''
        system = SISDynamicalSystem(N, X_init, A, param, cost)

        
        data_opt = []

        data_trivial = [] 
        data_trivial_frontld = []

        data_MN_degree_heuristic = []
        data_MN_frontld = []

        data_LN_degree_heuristic = []
        data_LN_frontld = []

        data_LSRS_heuristic = []

        data_CURE_heuristic = []

        data_MCM_heuristic = []

        for tr in range(trials_per_setting):
            print("\nTrial # " + str(tr))

           
            # stochastic optimal control
            data_opt.append(system.simulate_opt(time, plot=False))

            # Trivial heuristic (const.)
            data_trivial.append(system.simulate_trivial(
                setting['baselines']['trivial'], time, plot=False))

            # Trivial heuristic (front-loaded)

            data_trivial_frontld.append(
                system.simulate_trivial_frontloaded(
                    3,
                    setting['baselines']['frontld'],
                    setting['baselines']['trivial'],
                    time,
                    plot=False))
            

            # MN degree heuristic

            data_MN_degree_heuristic.append(system.simulate_MN_degree_heuristic(
                setting['baselines']['MN_deg'], time, plot=False))

            # MN deg heuristic (front-loaded)

            data_MN_frontld.append(
                system.simulate_MN_frontloaded(
                    3,
                    setting['baselines']['frontld'],
                    setting['baselines']['MN_deg'],
                    time,
                    plot=False))

            # LN degree heuristic
            data_LN_degree_heuristic.append(system.simulate_LN_degree_heuristic(
                setting['baselines']['LN_deg'], time, plot=False))

            # LN deg heuristic (front-loaded)

            data_LN_frontld.append(
                system.simulate_LN_frontloaded(
                    3,
                    setting['baselines']['frontld'],
                    setting['baselines']['LN_deg'],
                    time,
                    plot=False))

            # LRSR heuristic
            data_LSRS_heuristic.append(system.simulate_LRSR_heuristic(
                setting['baselines']['LRSR'], 
                time, plot=False))

            # CURE heuristic
            # data_CURE_heuristic.append(system.simulate_CURE_policy(
            #     setting['baselines']['frontld'],
            #     setting['baselines']['CURE'],
            #     time, plot=False))

            # MCM
            # data_MCM_heuristic.append(system.simulate_MCM(
            #     setting['baselines']['frontld'],
            #     setting['baselines']['MCM'],
            #     time, plot=False))




          


        '''Store data'''
        data = [
            data_opt, 

            data_trivial, 
            data_trivial_frontld,

            data_MN_degree_heuristic,
            data_MN_frontld,

            data_LN_degree_heuristic,
            data_LN_frontld,

            data_LSRS_heuristic,

            # data_CURE_heuristic,

            # data_MCM_heuristic,

        ]

        name = 'comparison_OPT_TR_MN_LN_LSRS_fair_i_'
        


        joblib.dump(data, 'temp_pickles/results_' + name 
                                                  + '_' 
                                                  + str(ind) 
                                                  + '_' 
                                                  + str(trials_per_setting) 
                                                  + '__Q_{}_{}'.format(int(round(setting['cost']['Qlam'][0])), 
                                                                       int(round(setting['cost']['Qx'][0])))
                                                  + '_.pkl')  





# 50 simulation run of all Qx settings

# settings = [
#     {'param': {
#         'beta':  6.0,
#         'gamma': 5.0,
#         'delta': 1.0,
#         'rho':   5.0,
#         'eta':   1.0},
#      'cost': {
#         'Qlam': 1.0 * np.ones(N),
#         'Qx': 1.0 * np.ones(N)},
#      'time': {
#         'total': 10.00
#     },
#         'baselines': {
#         'trivial': 0.35,
#         'MN_deg': 0.09,
#         'LN_deg': 1.0,  # TODO
#         'LRSR': 1.0,  # TODO
#         'CURE': 1.0,  # TODO
#         'MCM': 1.0,  # TODO
#         'frontld': dict(N=163.3000, peak_OPT=18.7468)
#     }
#     },
#     {'param': {
#         'beta':  6.0,
#         'gamma': 5.0,
#         'delta': 1.0,
#         'rho':   5.0,
#         'eta':   1.0},
#      'cost': {
#         'Qlam': 1.0 * np.ones(N),
#         'Qx': 10.0 * np.ones(N)},
#      'time': {
#         'total': 10.00
#     },
#         'baselines': {
#         'trivial': 0.25,
#         'MN_deg': 0.06,
#         'LN_deg': 1.0,  # TODO
#         'LRSR': 1.0,  # TODO
#         'CURE': 1.0,  # TODO
#         'MCM': 1.0,  # TODO
#         'frontld': dict(N=719.4000, peak_OPT=100.0833)
#     }
#     },
#     {'param': {
#         'beta':  6.0,
#         'gamma': 5.0,
#         'delta': 1.0,
#         'rho':   5.0,
#         'eta':   1.0},
#      'cost': {
#         'Qlam': 1.0 * np.ones(N),
#         'Qx': 20.0 * np.ones(N)},
#      'time': {
#         'total': 10.00
#     },
#         'baselines': {
#         'trivial': 0.2,
#         'MN_deg': 0.045,
#         'LN_deg': 1.0,  # TODO
#         'LRSR': 1.0,  # TODO
#         'CURE': 1.0,  # TODO
#         'MCM': 1.0,  # TODO
#         'frontld': dict(N=881.1667, peak_OPT=138.0000)
#     }
#     },
#     {'param': {
#         'beta':  6.0,
#         'gamma': 5.0,
#         'delta': 1.0,
#         'rho':   5.0,
#         'eta':   1.0},
#      'cost': {
#         'Qlam': 1.0 * np.ones(N),
#         'Qx': 50.0 * np.ones(N)},
#      'time': {
#         'total': 10.00
#     },
#         'baselines': {
#         'trivial': 0.1,
#         'MN_deg': 0.03,
#         'LN_deg': 1.0,  # TODO
#         'LRSR': 1.0,  # TODO
#         'CURE': 1.0,  # TODO
#         'MCM': 1.0,  # TODO
#         'frontld': dict(N=991.1000, peak_OPT=192.4839)
#     }
#     },
#     {'param': {
#         'beta':  6.0,
#         'gamma': 5.0,
#         'delta': 1.0,
#         'rho':   5.0,
#         'eta':   1.0},
#      'cost': {
#         'Qlam': 1.0 * np.ones(N),
#         'Qx': 100.0 * np.ones(N)},
#      'time': {
#         'total': 10.00
#     },
#         'baselines': {
#         'trivial': 0.035,
#         'MN_deg': 0.01,
#         'LN_deg': 1.0,  # TODO
#         'LRSR': 1.0,  # TODO
#         'CURE': 1.0,  # TODO
#         'MCM': 1.0,  # TODO
#         'frontld': dict(N=911.0333, peak_OPT=229.0777)
#     }
#     },
#     {'param': {
#         'beta':  6.0,
#         'gamma': 5.0,
#         'delta': 1.0,
#         'rho':   5.0,
#         'eta':   1.0},
#      'cost': {
#         'Qlam': 1.0 * np.ones(N),
#         'Qx': 150.0 * np.ones(N)},
#      'time': {
#         'total': 10.00
#     },
#         'baselines': {
#         'trivial': 0.018,
#         'MN_deg': 0.004,
#         'LN_deg': 1.0,  # TODO
#         'LRSR': 1.0,  # TODO
#         'CURE': 1.0,  # TODO
#         'MCM': 1.0,  # TODO
#         'frontld': dict(N=732.8000, peak_OPT=239.8862)
#     }
#     },
#     {'param': {
#         'beta':  6.0,
#         'gamma': 5.0,
#         'delta': 1.0,
#         'rho':   5.0,
#         'eta':   1.0},
#      'cost': {
#         'Qlam': 1.0 * np.ones(N),
#         'Qx': 200.0 * np.ones(N)},
#      'time': {
#         'total': 10.00
#     },
#         'baselines': {
#         'trivial': 0.01,
#         'MN_deg': 0.0025,
#         'LN_deg': 1.0,  # TODO
#         'LRSR': 1.0,  # TODO
#         'CURE': 1.0,  # TODO
#         'MCM': 1.0,  # TODO
#         'frontld': dict(N=615.8667, peak_OPT=241.1438)
#     }
#     },
#     {'param': {
#         'beta':  6.0,
#         'gamma': 5.0,
#         'delta': 1.0,
#         'rho':   5.0,
#         'eta':   1.0},
#      'cost': {
#         'Qlam': 1.0 * np.ones(N),
#         'Qx': 300.0 * np.ones(N)},
#      'time': {
#         'total': 10.00
#     },
#         'baselines': {
#         'trivial': 0.003,
#         'MN_deg': 0.0007,
#         'LN_deg': 1.0,  # TODO
#         'LRSR': 1.0,  # TODO
#         'CURE': 1.0,  # TODO
#         'MCM': 1.0,  # TODO
#         'frontld': dict(N=356.1000, peak_OPT=240.9349)
#     }
#     },
#     {'param': {
#         'beta':  6.0,
#         'gamma': 5.0,
#         'delta': 1.0,
#         'rho':   5.0,
#         'eta':   1.0},
#      'cost': {
#         'Qlam': 1.0 * np.ones(N),
#         'Qx': 400.0 * np.ones(N)},
#      'time': {
#         'total': 10.00
#     },
#         'baselines': {
#         'trivial': 0.0017,
#         'MN_deg': 0.0004,
#         'LN_deg': 1.0,  # TODO
#         'LRSR': 1.0,  # TODO
#         'CURE': 1.0,  # TODO
#         'MCM': 1.0,  # TODO
#         'frontld': dict(N=280.0331, peak_OPT=253.9399)
#     }
#     },
#     {'param': {
#         'beta':  6.0,
#         'gamma': 5.0,
#         'delta': 1.0,
#         'rho':   5.0,
#         'eta':   1.0},
#      'cost': {
#         'Qlam': 1.0 * np.ones(N),
#         'Qx': 500.0 * np.ones(N)},
#      'time': {
#         'total': 10.00
#     },
#         'baselines': {
#         'trivial': 0.00077,
#         'MN_deg': 0.00018,
#         'LN_deg': 1.0,  # TODO
#         'LRSR': 1.0,  # TODO
#         'CURE': 1.0,  # TODO
#         'MCM': 1.0,  # TODO
#         'frontld': dict(N=166.6667, peak_OPT=287.7502)
#     }
#     },

# ]



# LN_LRSR_ii

# settings = [
#     {'param': {
#         'beta':  6.0,
#         'gamma': 5.0,
#         'delta': 1.0,
#         'rho':   5.0,
#         'eta':   1.0},
#      'cost': {
#         'Qlam': 1.0 * np.ones(N),
#         'Qx': 1.0 * np.ones(N)},
#      'time': {
#         'total': 10.00
#     },
#         'baselines': {
#         'trivial': 0.35,
#         'MN_deg': 0.09,
#         'LN_deg': 0.10,  # TODO
#         'LRSR': 4.0,  # TODO
#         'CURE': 1.0,  # TODO
#         'MCM': 1.0,  # TODO
#         'frontld': dict(N=163.3000, peak_OPT=18.7468)
#     }
#     },
#     {'param': {
#         'beta':  6.0,
#         'gamma': 5.0,
#         'delta': 1.0,
#         'rho':   5.0,
#         'eta':   1.0},
#      'cost': {
#         'Qlam': 1.0 * np.ones(N),
#         'Qx': 10.0 * np.ones(N)},
#      'time': {
#         'total': 10.00
#     },
#         'baselines': {
#         'trivial': 0.25,
#         'MN_deg': 0.06,
#         'LN_deg': 0.10,  # TODO
#         'LRSR': 4.0,  # TODO
#         'CURE': 1.0,  # TODO
#         'MCM': 1.0,  # TODO
#         'frontld': dict(N=719.4000, peak_OPT=100.0833)
#     }
#     },
#     {'param': {
#         'beta':  6.0,
#         'gamma': 5.0,
#         'delta': 1.0,
#         'rho':   5.0,
#         'eta':   1.0},
#      'cost': {
#         'Qlam': 1.0 * np.ones(N),
#         'Qx': 20.0 * np.ones(N)},
#      'time': {
#         'total': 10.00
#     },
#         'baselines': {
#         'trivial': 0.2,
#         'MN_deg': 0.045,
#         'LN_deg': 0.06,  # TODO
#         'LRSR': 4.0,  # TODO
#         'CURE': 1.0,  # TODO
#         'MCM': 1.0,  # TODO
#         'frontld': dict(N=881.1667, peak_OPT=138.0000)
#     }
#     },
#     {'param': {
#         'beta':  6.0,
#         'gamma': 5.0,
#         'delta': 1.0,
#         'rho':   5.0,
#         'eta':   1.0},
#      'cost': {
#         'Qlam': 1.0 * np.ones(N),
#         'Qx': 50.0 * np.ones(N)},
#      'time': {
#         'total': 10.00
#     },
#         'baselines': {
#         'trivial': 0.1,
#         'MN_deg': 0.03,
#         'LN_deg': 0.04,  # TODO
#         'LRSR': 4.0,  # TODO
#         'CURE': 1.0,  # TODO
#         'MCM': 1.0,  # TODO
#         'frontld': dict(N=991.1000, peak_OPT=192.4839)
#     }
#     },
#     {'param': {
#         'beta':  6.0,
#         'gamma': 5.0,
#         'delta': 1.0,
#         'rho':   5.0,
#         'eta':   1.0},
#      'cost': {
#         'Qlam': 1.0 * np.ones(N),
#         'Qx': 100.0 * np.ones(N)},
#      'time': {
#         'total': 10.00
#     },
#         'baselines': {
#         'trivial': 0.035,
#         'MN_deg': 0.015,
#         'LN_deg': 0.015,  # TODO
#         'LRSR': 4.0,  # TODO
#         'CURE': 1.0,  # TODO
#         'MCM': 1.0,  # TODO
#         'frontld': dict(N=911.0333, peak_OPT=229.0777)
#     }
#     },
#     {'param': {
#         'beta':  6.0,
#         'gamma': 5.0,
#         'delta': 1.0,
#         'rho':   5.0,
#         'eta':   1.0},
#      'cost': {
#         'Qlam': 1.0 * np.ones(N),
#         'Qx': 150.0 * np.ones(N)},
#      'time': {
#         'total': 10.00
#     },
#         'baselines': {
#         'trivial': 0.018,
#         'MN_deg': 0.004,
#         'LN_deg': 0.004,  # TODO
#         'LRSR': 4.0,  # TODO
#         'CURE': 1.0,  # TODO
#         'MCM': 1.0,  # TODO
#         'frontld': dict(N=732.8000, peak_OPT=239.8862)
#     }
#     },
#     {'param': {
#         'beta':  6.0,
#         'gamma': 5.0,
#         'delta': 1.0,
#         'rho':   5.0,
#         'eta':   1.0},
#      'cost': {
#         'Qlam': 1.0 * np.ones(N),
#         'Qx': 200.0 * np.ones(N)},
#      'time': {
#         'total': 10.00
#     },
#         'baselines': {
#         'trivial': 0.01,
#         'MN_deg': 0.0030,
#         'LN_deg': 0.0030,  # TODO
#         'LRSR': 4.0,  # TODO
#         'CURE': 1.0,  # TODO
#         'MCM': 1.0,  # TODO
#         'frontld': dict(N=615.8667, peak_OPT=241.1438)
#     }
#     },
#     {'param': {
#         'beta':  6.0,
#         'gamma': 5.0,
#         'delta': 1.0,
#         'rho':   5.0,
#         'eta':   1.0},
#      'cost': {
#         'Qlam': 1.0 * np.ones(N),
#         'Qx': 300.0 * np.ones(N)},
#      'time': {
#         'total': 10.00
#     },
#         'baselines': {
#         'trivial': 0.003,
#         'MN_deg': 0.0008,
#         'LN_deg': 0.0008,  # TODO
#         'LRSR': 4.0,  # TODO
#         'CURE': 1.0,  # TODO
#         'MCM': 1.0,  # TODO
#         'frontld': dict(N=356.1000, peak_OPT=240.9349)
#     }
#     },
#     {'param': {
#         'beta':  6.0,
#         'gamma': 5.0,
#         'delta': 1.0,
#         'rho':   5.0,
#         'eta':   1.0},
#      'cost': {
#         'Qlam': 1.0 * np.ones(N),
#         'Qx': 400.0 * np.ones(N)},
#      'time': {
#         'total': 10.00
#     },
#         'baselines': {
#         'trivial': 0.0017,
#         'MN_deg': 0.0004,
#         'LN_deg': 0.0004,  # TODO
#         'LRSR': 4.0,  # TODO
#         'CURE': 1.0,  # TODO
#         'MCM': 1.0,  # TODO
#         'frontld': dict(N=280.0331, peak_OPT=253.9399)
#     }
#     },
#     {'param': {
#         'beta':  6.0,
#         'gamma': 5.0,
#         'delta': 1.0,
#         'rho':   5.0,
#         'eta':   1.0},
#      'cost': {
#         'Qlam': 1.0 * np.ones(N),
#         'Qx': 500.0 * np.ones(N)},
#      'time': {
#         'total': 10.00
#     },
#         'baselines': {
#         'trivial': 0.00077,
#         'MN_deg': 0.00018,
#         'LN_deg': 0.00018,  # TODO
#         'LRSR': 4.0,  # TODO
#         'CURE': 1.0,  # TODO
#         'MCM': 1.0,  # TODO
#         'frontld': dict(N=166.6667, peak_OPT=287.7502)
#     }
#     },

# ]
