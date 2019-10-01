from multiprocessing import Process, cpu_count
import argparse
import sys
import os

import script_single_job

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', dest='dir', type=str,
                        required=True, help="Experiment directory")
    parser.add_argument('-n', '--n_sims', dest='n_sims', type=int,
                        required=True, help="Number of simulatins per graph and per index")
    parser.add_argument('-p', '--pool', dest='n_workers', type=int,
                        required=False, default=cpu_count() - 1,
                        help="Size of the parallel pool")
    args = parser.parse_args()

    # Init the list of arguments for the workers
    pool_args = list()

    # Extract the list of parameter file in the experiment directory
    param_file_list = sorted([f for f in os.listdir(args.dir) if f.startswith('param')])

    # Make pool args
    for param_filename in param_file_list:
        for sim_idx in range(args.n_sims):
            # Extract suffix from param_filename
            suffix = '-'.join(param_filename.rstrip('.json').split('-')[1:])
            # Add simulation infex
            suffix += f'-sim{sim_idx:2>d}'
            # Build output filename
            output_filename = f'output-{suffix}.json'
            # Redirect stdout
            stdout = os.path.join(args.dir, f"stdout-{suffix:s}")
            # Redirect stderr
            stderr = os.path.join(args.dir, f"stderr-{suffix:s}")
            # Add script arguments to the pool list
            pool_args.append(
                (args.dir, param_filename, output_filename, stdout, stderr)
            )

    print(f"Start {len(pool_args):d} experiments on a pool of {args.n_workers:d} workers")
    print(f"=============================================================================")

    proc_list = list()

    while len(pool_args) > 0:

        this_args = pool_args.pop()

        print("Start process with parameters:", this_args)

        proc = Process(target=script_single_job.run, args=this_args)
        proc_list.append(proc)
        proc.start()

        if len(proc_list) == args.n_workers:
            # Wait until all processes are done
            for proc in proc_list:
                proc.join()
            # Reset process list
            proc_list = list()
            print()

    print('Done.')
