#
# Calibration of baseline control parameters
#

import os
import json
import copy
import itertools
from collections import Counter, defaultdict
import pandas as pd
import networkx as nx
import numpy as np
from multiprocessing import cpu_count, Pool

from lib.graph_generation import make_ebola_network
from lib.dynamics import SimulationSIR, PriorityQueue
from lib.dynamics import sample_seeds
from lib.settings import DATA_DIR, PROJECT_DIR
from lib import metrics


# 1. Set simulation parameters
# ============================

# Set simulation parameters
start_day_str = '2014-01-01'
end_day_str = '2014-04-01'
max_timedelta = pd.to_datetime(end_day_str) - pd.to_datetime(start_day_str)
max_days = max_timedelta.days

# Set SIR infection and recovery rates
beta = 1 / 15.3
delta = 1 / 11.4
gamma = beta
rho = 0.0

# Set the network parameters.
n_nodes = 8000
p_in = 0.01
p_out = {
    'Guinea': 0.00215,
    'Liberia': 0.00300,
    'Sierra Leone': 0.00315,
    'inter-country': 0.0019
}

# Set the control parameters.
DEFAULT_POLICY_PARAMS = {
    # SOC parameters
    'eta':   1.0,          # SOC exponential decay
    'q_x':   None,       # SOC infection cost
    'q_lam':   1.0,        # SOC recovery cost
    'lpsolver': 'cvxopt',  # SOC linear progam solver
    
    # Scaling of baseline methods
    'TR': None,
    'MN': None,
    'LN': None,
    'LRSR': None,
    'MCM': None,
    'front-loading': {  # Front-loading parameters (will be set after the SOC run)
        'max_interventions': None,
        'max_lambda': None
    }
}


# 2. Run calibration
# ==================


def worker(policy, policy_params, n_sims, q_idx, net_idx, output_filename):
    graph = make_ebola_network(n_nodes=n_nodes, p_in=p_in, p_out=p_out)
    print(f'graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges')

    init_event_list = sample_seeds(graph, delta=delta, method='data', 
                                   max_date=start_day_str, verbose=False)

    res_dict = {
        'max_u': list(), 
        'n_tre': list(), 
        'q_x': policy_params['q_x'],
        'q_idx': q_idx,
        'net_idx': net_idx,
    }

    for sim_idx in range(n_sims):
         
        sir_obj = SimulationSIR(graph, beta=beta, delta=delta, gamma=gamma, rho=rho, verbose=False)
        sir_obj.launch_epidemic(
            init_event_list=init_event_list,
            max_time=max_days, 
            policy=policy,
            policy_dict=policy_params
        )
        
        res_dict['max_u'].append(float(sir_obj.max_total_control_intensity))
        res_dict['n_tre'].append(float(sir_obj.is_tre.sum()))

        print(f"Finished: q_x:{q_idx} ({policy_params['q_x']}) net:{net_idx+1} sim:{sim_idx+1}/{n_sims}")
    
    with open(output_filename, 'w') as f:
        json.dump(res_dict, f)

if __name__ == "__main__":

    OUT_DIR = os.path.join(PROJECT_DIR, 'output', 'baseline-calibration-soc')
    if not os.path.exists(OUT_DIR):
        print(f"Create output directory: {OUT_DIR}")
        os.mkdir(OUT_DIR)

    Q_X_RANGE = [1.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 750.0, 1000.0]

    NUM_NETS = 5
    NUM_SIMS = 5

    args_list = list()
    for q_idx, q_x in enumerate(Q_X_RANGE):
        for net_idx in range(NUM_NETS):
            
            policy_params = copy.deepcopy(DEFAULT_POLICY_PARAMS)
            policy_params['q_x'] = q_x

            output_filename = os.path.join(OUT_DIR, f"output-q{q_idx:d}-n{net_idx:d}.json")

            args_list.append(('SOC', policy_params, NUM_SIMS, q_idx, net_idx, output_filename))
        
    n_procs = cpu_count()-1

    print(f"\nRun {len(args_list)} jobs on {n_procs} processes...\n")

    pool = Pool(n_procs)
    pool.starmap(worker, args_list)

