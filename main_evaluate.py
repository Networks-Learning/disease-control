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

from dynamics import SISDynamicalSystem
from analysis import Evaluation
from stochastic_processes import StochasticProcess, CountingProcess

plt.switch_backend('agg')

if __name__ == '__main__':


    '''Evaluate results'''
    names = [['Stochastic optimal control ',
              'MN Deg Heuristic (~ deg(v) Qx/Qlam (1-H) X)',
              'Trivial (~ 5 Qx/Qlam (1-H) X)'],
             ['Stochastic optimal control ',
              'MN Deg Heuristic (~ deg(v) Qx/Qlam (1-H) X)',
              'Trivial (~ Qx/Qlam (1-H) X)']]
    saved = {
        1: ('results_kREG_200_opt_deg_5triv.pkl', names[0]),
        2: ('results_FB_200_opt_deg_5triv.pkl',  names[0]),
        3: ('results_Qs_0__US_30_opt_deg_triv.pkl',  names[1]),
        4: ('results_Qs_1__US_30_opt_deg_triv.pkl',  names[1]),
        5: ('results_Qs_2__US_30_opt_deg_triv.pkl',  names[1]),
        6: ('results_Qs_3__US_30_opt_deg_triv.pkl',  names[1]),
        7: ('results_Qs_4__US_30_opt_deg_triv.pkl',  names[1]),
        8: ('results_etas_0__US_30_opt_deg_triv.pkl',  names[1]),
        9: ('results_etas_1__US_30_opt_deg_triv.pkl',  names[1]),
        10: ('results_etas_2__US_30_opt_deg_triv.pkl',  names[1]),
    }
    all_selected = [3, 4, 5, 6, 7, 8, 9, 10] # select pickle files to import

    for selected in all_selected:
        data = joblib.load('temp_pickles/' + saved[selected][0])
        filename = saved[selected][0]
        description = saved[selected][1]
        eval = Evaluation(data, filename, description)


        # to see graphs instead of saving them, comment out 'plt.switch_backend('agg')' from top of main.py
        # do not comment out when running on cluster

        print("Starting evaluation...\n")


        eval.present_discounted_loss(plot=True, save=True)
        eval.present_discounted_loss_BOX(plot=True, save=True)

        eval.simulation_treatment_plot(granularity=0.001, save=True)
        eval.simulation_infection_plot(granularity=0.001, save=True)

        eval.infection_cost_AND_total_interventions(plot=True, save=True)
        eval.infection_cost_AND_time_under_treatment(plot=True, save=True)
        eval.infection_cost_AND_intervention_effort(plot=True, save=True)
        eval.infection_cost_per_time_under_treatment(plot=True, save=True)

        print("\n... evaluation finished.")
