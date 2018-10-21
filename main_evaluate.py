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
import collections

from dynamics import SISDynamicalSystem
from analysis import Evaluation, MultipleEvaluations
from stochastic_processes import StochasticProcess, CountingProcess

plt.switch_backend('agg')

if __name__ == '__main__':


    '''Evaluate results'''
    names = [['Stochastic OPT ',
              'Tr',
              'Tr FL (1)',
              'Tr FL (2)',
              'Tr FL (3)',
              'MN',
              'MN FL (1)',
              'MN FL (2)',
              'MN FL (3)'],
             ['Stochastic OPT ',
              'Tr',
              'Tr FL (1)',
              'Tr FL (2)',
              'Tr FL (3)']]



    saved = {
        -1: ('results_comparison_OPT_T_MN_debuggingfile_0_30__Q_1_100_.pkl', names[1]),
        0: ('results_comparison_OPT_T_MN_complete_0_50__Q_1_1_.pkl', names[0]),
        1: ('results_comparison_OPT_T_MN_complete_1_50__Q_1_10_.pkl', names[0]),
        2: ('results_comparison_OPT_T_MN_complete_2_50__Q_1_20_.pkl', names[0]),
        3: ('results_comparison_OPT_T_MN_complete_3_50__Q_1_50_.pkl', names[0]),
        4: ('results_comparison_OPT_T_MN_complete_4_50__Q_1_100_.pkl', names[0]),
        5: ('results_comparison_OPT_T_MN_complete_5_50__Q_1_150_.pkl', names[0]),
        6: ('results_comparison_OPT_T_MN_complete_6_50__Q_1_200_.pkl', names[0]),
        7: ('results_comparison_OPT_T_MN_complete_7_50__Q_1_300_.pkl', names[0]),
        8: ('results_comparison_OPT_T_MN_complete_8_50__Q_1_400_.pkl', names[0]),
        9: ('results_comparison_OPT_T_MN_complete_9_50__Q_1_500_.pkl', names[0]),
        
    }

    # constant interventions with Qx varying              
    #               1  10 20 50 100 150 200 300 400 500
    all_selected = [0, 1, 2, 3, 4,  5,  6,  7,  8,  9]


    # all_selected = [-1, 8]  # select pickle files to import


    # summary for multi setting comparison
    multi_summary = collections.defaultdict(dict)

    '''Individual analyses'''
    for selected in all_selected:

        # to see graphs instead of saving them, comment out 'plt.switch_backend('agg')' from top of main.py
        # do not comment out when running on cluster


        print('Analyzing:  {}'.format(saved[selected][0]))

        data = joblib.load('temp_pickles/' + saved[selected][0])
        filename = saved[selected][0]
        description = saved[selected][1]
        eval = Evaluation(data, filename, description)

        multi_summary['eval_obj'][saved[selected][0]] = eval

       

        # Individual analysis

        # eval.infections_and_interventions_complete(save=True)
        # eval.simulation_infection_plot(granularity=0.001, save=True)
        # eval.simulation_treatment_plot(granularity=0.001, save=True)
        # eval.present_discounted_loss(plot=True, save=True)




        ''''''''''''''''''''''''''''''''''''''''''
        
        # Comparison analysis 

        summary_tup = eval.infections_and_interventions_complete(size_tup = (13, 8), save=True)
        multi_summary['infections_and_interventions'][saved[selected][0]] = summary_tup

        # summary_tup = eval.summarize_interventions_and_intensities()
        # multi_summary['stats_intervention_intensities'][saved[selected][0]] = summary_tup


        ''''''''''''''''''''''''''''''''''''''''''
        
        # eval.debug()


    '''Comparative analysis'''
    multi_eval = MultipleEvaluations(saved, all_selected, multi_summary)

    multi_eval.compare_infections(size_tup=(7, 5), save=True)
