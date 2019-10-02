import random
import numpy as np
import pandas as pd
import networkx as nx
import argparse
import os
import json
import sys

from lib.graph_generation import make_ebola_network
from lib.dynamics import SimulationSIR, PriorityQueue
from lib.dynamics import sample_seeds
from lib.settings import DATA_DIR


def run(exp_dir, param_filename, output_filename, stdout=None, stderr=None, verbose=False):
    """
    Run a single SIR simulation based on the parameters in `param_filename` inside directory
    `exp_dir` and output a summary into `output_filename`.

    stdout and stderr can be redirected to files `stdout` and `stderr`.

    The parameter file is supposed to be a json file with the following format:
    ```
    {
        network : { (Network model parameters)
            n_nodes : int. Number of nodes desired in the network
            p_in : float. Intra-district probability
            p_out : float. Inter-district probability
            seed : int. Random seed for reproducibility
        },
        simulation : { (SIR simulation parameters)
            start_day_str : str. Starting day of the simulation. Used to sample seeds from the 
                Ebola dataset. Formated as 'YYYY-MM-DD'.
            end_day_str : str. Ending day of the simulation.
            sir_params : { (Parameters of the SIR model)
                beta : float. Infection rate
                delta : float. Recovery rate (spontaneous)
                gamma : float. Reduction of infectivity under treatement
                rho : float. ecovery rate under treatement
            },
            policy_name : str. Name of the Policy.
            policy_params : { (Parameters of the Policy)
                (depend on the policy)
            }
        }
    }
    ```

    The output file contains:
    - the intial seed events,
    - the infection time of each node,
    - the infector of each node,
    - the recovery time of each node,
    - the district of each node.

    """

    if stdout is not None:
        sys.stdout = open(stdout, 'w')
    if stderr is not None:
        sys.stderr = open(stderr, 'w')

    # Load parameters from file
    param_filename_full = os.path.join(exp_dir, param_filename)
    if not os.path.exists(param_filename_full):
        raise FileNotFoundError('Input file `{:s}` not found.'.format(param_filename_full))
    with open(param_filename_full, 'r') as param_file:
        param_dict = json.load(param_file)

    print('\nExperiment parameters')
    print('=====================')
    print(f'        exp_dir = {exp_dir:s}')
    print(f' param_filename = {param_filename:s}')
    print(f'output_filename = {output_filename:s}')

    # Init output dict
    output_dict = {}

    # Generate network of districts
    # =============================

    print('\nGENERATE NETWORK')
    print('================')

    print('\nNetwork parameters')
    print(f"  - n_nodes = {param_dict['network']['n_nodes']:d}")
    print(f"  - p_in = {param_dict['network']['p_in']:.2e}")
    print(f"  - p_out = {param_dict['network']['p_out']:.2e}")
    print(f"  - seed = {param_dict['network']['seed']}")

    graph = make_ebola_network(**param_dict['network'])

    print('\nGraph generated')
    print(f"  - {graph.number_of_nodes():d} nodes")
    print(f"  - {graph.number_of_edges():d} edges")

    # Run simulation
    # ==============

    print('\nSIMULATION')
    print('==========')

    start_day_str = param_dict['simulation']['start_day_str']
    end_day_str = param_dict['simulation']['end_day_str']
    sim_timedelta = pd.to_datetime(end_day_str) - pd.to_datetime(start_day_str)
    max_time = sim_timedelta.days

    print('\nSimulation parameters')
    print(f'  - start day: {start_day_str}')
    print(f'  -   end day: {start_day_str}')
    print(f'  - number of days to simulate: {max_time}')
    
    print('\nEpidemic parameters')
    for key, val in param_dict['simulation']['sir_params'].items():
        print(f'  - {key:s}: {val:.2e}')
    
    print(f"\nPolicy name: {param_dict['simulation']['policy_name']:s}")
    
    print('\nPolicy parameters')
    for key, val in param_dict['simulation']['policy_params'].items():
        print(f'  - {key:s}: {val:.2e}')
    
    # Reinitialize random seed for simulation
    random.seed(None)
    seed = random.randint(0, 2**32-1)
    random.seed(seed)
    print(f'Random seed: {seed}')
    # Add to output dict
    output_dict['simulation_seed'] = seed

    # Sample initial infected seeds at time t=0
    delta = param_dict['simulation']['sir_params']['delta']
    init_event_list = sample_seeds(graph, delta=delta, max_date=start_day_str, verbose=verbose)

    print('\nRun simulation...')
    
    # Run SIR simulation
    sir_obj = SimulationSIR(graph, **param_dict['simulation']['sir_params'], verbose=verbose)
    sir_obj.launch_epidemic(init_event_list=init_event_list, max_time=max_time,
                            policy=param_dict['simulation']['policy_name'],
                            policy_dict=param_dict['simulation']['policy_params'])

    # Post-simulation summarization and output
    # ========================================

    print('\nPOST-SIMULATION')
    print('===============')

    # Add init_event_list to output dict
    # Format init_event_list node names into int to make the object json-able
    for i, (e, t) in enumerate(init_event_list):
        init_event_list[i] = ((int(e[0]), e[1], int(e[2]) if e[2] is not None else None), float(t))
    output_dict['init_event_list'] = init_event_list

    # Add other info on the events of each node
    output_dict['inf_occured_at'] = sir_obj.inf_occured_at.tolist()
    output_dict['rec_occured_at'] = sir_obj.rec_occured_at.tolist()
    output_dict['infector'] = sir_obj.infector.tolist()

    country_list = np.zeros(sir_obj.n_nodes, dtype=object)
    for u, d in sir_obj.G.nodes(data=T):
        country_list[sir_obj.node_to_idx[u]] = d['country']
    output_dict['country'] = country_list

    node_district_arr = np.zeros(sir_obj.n_nodes, dtype='object')
    for node, data in sir_obj.G.nodes(data=True):
        node_idx = sir_obj.node_to_idx[node]
        node_district_arr[node_idx] = data['district']
    output_dict['district'] = node_district_arr.tolist()

    print('\nSave results...')

    with open(os.path.join(exp_dir, output_filename), 'w') as output_file:
        json.dump(output_dict, output_file)

    # Log that the run is finished
    print('\n\nFinished.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', dest='dir', type=str,
                        required=True, help="Working directory")
    parser.add_argument('-p', '--params', dest='param_filename', type=str,
                        required=False, default='params.json',
                        help="Input parameter file (JSON)")
    parser.add_argument('-o', '--outfile', dest='output_filename', type=str,
                        required=False, default='output.json',
                        help="Output file (JSON)")
    parser.add_argument('-v', '--verbose', dest='verbose', action="store_true",
                        required=False, default=False,
                        help="Print behavior")
    args = parser.parse_args()

    run(exp_dir=args.dir, param_filename=args.param_filename, output_filename=args.output_filename,
        verbose=args.verbose)
