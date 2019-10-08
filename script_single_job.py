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


def run(exp_dir, param_filename, output_filename, net_idx, stdout=None, stderr=None, verbose=False):
    """
    Run a single SIR simulation based on the parameters in `param_filename` inside directory
    `exp_dir` and output a summary into `output_filename`. Uses the random seed at index `net_idx`
    to generate a network.

    stdout and stderr can be redirected to files `stdout` and `stderr`.

    The parameter file is supposed to be a json file with the following format:
    ```
    {
        
        network : { (Network model parameters)
            n_nodes : int
                Number of nodes desired in the network
            p_in : float
                Intra-district probability
            p_out : dict of float
                Inter-district probability per country, keyed by country name, with the additional
                key 'inter-country' for between-country edges
            seed_list : list. List of random seeds for reproducibility
        },
        
        simulation : { (SIR simulation parameters)
            start_day_str : str
                Starting day of the simulation, formated as 'YYYY-MM-DD'. Used to sample seeds
                from the Ebola dataset.
            end_day_str : str
                Ending day of the simulation.
            sir_params : { (Parameters of the SIR model)
                beta : float
                    Infection rate
                delta : float
                    Recovery rate (spontaneous)
                gamma : float
                    Reduction of infectivity under treatement
                rho : float
                    Recovery rate under treatement
            },
            policy_name : str
                Name of the Policy.
            policy_params : { (Parameters of the Policy)
                (depend on the policy)
            }
        },
        
        job_type : str (optional)
            One the predefined job types. By default, perform a standard simulation following all
            the given parameters. Other job types are the following:
            - 'stop_after_seeds':
                Only perform the simulation on the seeds ego-network and stop once all seeds are
                recovered or once their neighbors are all infected. This job is performed to assess
                the basic reproduction number of the epidemic given the current parameters.

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
    print(f'output_filename = {output_filename:s}', flush=True)

    # Init output dict
    output_dict = {}

    # Generate network of districts
    # =============================

    print('\nGENERATE NETWORK')
    print('================')

    # Extract random seed from list
    net_seed = param_dict['network']['seed_list'][net_idx]
    param_dict['network']['seed'] = net_seed
    del param_dict['network']['seed_list']

    print('\nNetwork parameters')
    print(f"  - n_nodes = {param_dict['network']['n_nodes']:d}")
    print(f"  - p_in = {param_dict['network']['p_in']:.2e}")
    print(f"  - p_out = {param_dict['network']['p_out']}")
    print(f"  - seed = {net_seed:d}")

    graph = make_ebola_network(**param_dict['network'])

    print('\nGraph generated')
    print(f"  - {graph.number_of_nodes():d} nodes")
    print(f"  - {graph.number_of_edges():d} edges", flush=True)

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
    print(f'  -   end day: {end_day_str}')
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
    init_seed_method = param_dict['simulation']['init_seed_method']
    init_event_list = sample_seeds(graph, delta=delta, method=init_seed_method,
                                   max_date=start_day_str, verbose=verbose)

    # Set default stopping criteria
    stop_criteria = None

    # Modify parameters for special job types
    if param_dict.get('job_type') == 'stop_after_seeds':
        # Extract ego-network of seeds
        seed_node_list = np.array(list(set([event[0] for event, _ in init_event_list])))
        seed_neighbs_list = np.hstack([list(graph.neighbors(u)) for u in seed_node_list])
        graph = nx.subgraph(graph, np.hstack((seed_node_list, seed_neighbs_list)))
        
        # Define stop_criteria
        def stop_criteria(sir_obj):
            seed_node_indices = np.array([sir_obj.node_to_idx[u] for u in seed_node_list])
            seed_neighbs_indices = np.array([sir_obj.node_to_idx[u] for u in seed_neighbs_list])
            all_seeds_rec = np.all(sir_obj.is_rec[seed_node_indices])
            all_neighbors_inf = np.all(sir_obj.is_inf[seed_neighbs_indices])
            return all_seeds_rec or all_neighbors_inf

    print('\nRun simulation...', flush=True)
    
    # Run SIR simulation
    sir_obj = SimulationSIR(graph, **param_dict['simulation']['sir_params'], verbose=verbose)
    sir_obj.launch_epidemic(init_event_list=init_event_list, max_time=max_time,
                            policy=param_dict['simulation']['policy_name'],
                            policy_dict=param_dict['simulation']['policy_params'],
                            stop_criteria=stop_criteria
                            )

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
    output_dict['node_idx_pairs'] = [(int(u), u_idx) for u, u_idx in sir_obj.node_to_idx.items()]

    country_list = np.zeros(sir_obj.n_nodes, dtype=object)
    for u, d in sir_obj.G.nodes(data=True):
        country_list[sir_obj.node_to_idx[u]] = d['country']
    output_dict['country'] = country_list.tolist()

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
    
    parser.add_argument('-i', '--netidx', dest='net_idx', type=int,
                        required=True, help="Network index to use (in parameter file)")

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

    run(
        exp_dir=args.dir,
        param_filename=args.param_filename,
        output_filename=args.output_filename,
        net_idx=args.net_idx,
        verbose=args.verbose
    )
