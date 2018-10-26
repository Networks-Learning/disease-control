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
              'Tr FL (3)'],
             ['Stochastic OPT ',
              'LN',
              'LN FL (3)',
              'LSRS'],
             ['SOC',
              'T',
              'T-FL',
              'MN',
              'MN-FL',
              'LN',
              'LN-FL',
              'LRSR']]


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

        10: ('results_comparison_OPT_LN_LSRS_0_30__Q_1_1_.pkl', names[2]),
        11: ('results_comparison_OPT_LN_LSRS_1_30__Q_1_10_.pkl', names[2]),
        12: ('results_comparison_OPT_LN_LSRS_2_30__Q_1_20_.pkl', names[2]),
        13: ('results_comparison_OPT_LN_LSRS_3_30__Q_1_50_.pkl', names[2]),
        14: ('results_comparison_OPT_LN_LSRS_4_30__Q_1_100_.pkl', names[2]),
        15: ('results_comparison_OPT_LN_LSRS_5_30__Q_1_150_.pkl', names[2]),
        16: ('results_comparison_OPT_LN_LSRS_6_30__Q_1_200_.pkl', names[2]),
        17: ('results_comparison_OPT_LN_LSRS_7_30__Q_1_300_.pkl', names[2]),
        18: ('results_comparison_OPT_LN_LSRS_8_30__Q_1_400_.pkl', names[2]),
        19: ('results_comparison_OPT_LN_LSRS_9_30__Q_1_500_.pkl', names[2]),
        
        20: ('results_comparison_OPT_LN_LSRS_ii__0_30__Q_1_1_.pkl', names[2]),
        21: ('results_comparison_OPT_LN_LSRS_ii__1_30__Q_1_10_.pkl', names[2]),
        22: ('results_comparison_OPT_LN_LSRS_ii__2_30__Q_1_20_.pkl', names[2]),
        23: ('results_comparison_OPT_LN_LSRS_ii__3_30__Q_1_50_.pkl', names[2]),
        24: ('results_comparison_OPT_LN_LSRS_ii__4_30__Q_1_100_.pkl', names[2]),
        25: ('results_comparison_OPT_LN_LSRS_ii__5_30__Q_1_150_.pkl', names[2]),
        26: ('results_comparison_OPT_LN_LSRS_ii__6_30__Q_1_200_.pkl', names[2]),
        27: ('results_comparison_OPT_LN_LSRS_ii__7_30__Q_1_300_.pkl', names[2]),
        28: ('results_comparison_OPT_LN_LSRS_ii__8_30__Q_1_400_.pkl', names[2]),
        29: ('results_comparison_OPT_LN_LSRS_ii__9_30__Q_1_500_.pkl', names[2]),

        30: ('results_comparison_OPT_LN_LSRS_iii__0_30__Q_1_1_.pkl', names[2]),
        31: ('results_comparison_OPT_LN_LSRS_iii__1_30__Q_1_10_.pkl', names[2]),
        32: ('results_comparison_OPT_LN_LSRS_iii__2_30__Q_1_20_.pkl', names[2]),
        33: ('results_comparison_OPT_LN_LSRS_iii__3_30__Q_1_50_.pkl', names[2]),
        34: ('results_comparison_OPT_LN_LSRS_iii__4_30__Q_1_100_.pkl', names[2]),
        35: ('results_comparison_OPT_LN_LSRS_iii__5_30__Q_1_150_.pkl', names[2]),
        36: ('results_comparison_OPT_LN_LSRS_iii__6_30__Q_1_200_.pkl', names[2]),
        37: ('results_comparison_OPT_LN_LSRS_iii__7_30__Q_1_300_.pkl', names[2]),
        38: ('results_comparison_OPT_LN_LSRS_iii__8_30__Q_1_400_.pkl', names[2]),
        39: ('results_comparison_OPT_LN_LSRS_iii__9_30__Q_1_500_.pkl', names[2]),

        40: ('results_comparison_OPT_TR_MN_LN_LSRS_fair_i__0_50__Q_1_1_.pkl', names[3]),
        41: ('results_comparison_OPT_TR_MN_LN_LSRS_fair_i__1_50__Q_1_10_.pkl', names[3]),
        42: ('results_comparison_OPT_TR_MN_LN_LSRS_fair_i__2_50__Q_1_20_.pkl', names[3]),
        43: ('results_comparison_OPT_TR_MN_LN_LSRS_fair_i__3_50__Q_1_50_.pkl', names[3]),
        44: ('results_comparison_OPT_TR_MN_LN_LSRS_fair_i__4_50__Q_1_100_.pkl', names[3]),
        45: ('results_comparison_OPT_TR_MN_LN_LSRS_fair_i__5_50__Q_1_150_.pkl', names[3]),
        46: ('results_comparison_OPT_TR_MN_LN_LSRS_fair_i__6_50__Q_1_200_.pkl', names[3]),
        47: ('results_comparison_OPT_TR_MN_LN_LSRS_fair_i__7_50__Q_1_300_.pkl', names[3]),
        48: ('results_comparison_OPT_TR_MN_LN_LSRS_fair_i__8_50__Q_1_400_.pkl', names[3]),
        49: ('results_comparison_OPT_TR_MN_LN_LSRS_fair_i__9_50__Q_1_500_.pkl', names[3]),
    }         

    # constant interventions with Qx varying              

    #               1   10  20  50  100 150 200 300 400 500
    all_selected = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]

    all_selected = [48]  # select pickle files to import

    multi_summary_from_dump = False

    # summary for multi setting comparison
    multi_summary = collections.defaultdict(dict)

    if not multi_summary_from_dump:

        '''Individual analyses'''
        for selected in all_selected:

            # to see graphs instead of saving them, comment out 'plt.switch_backend('agg')' from top of main.py
            # do not comment out when running on cluster


            print('Analyzing:  {}'.format(saved[selected][0]))

            data = joblib.load('temp_pickles/' + saved[selected][0])
            filename = saved[selected][0]
            description = saved[selected][1]
            eval = Evaluation(data, filename, description)

            multi_summary['Qs'][saved[selected][0]] = eval.data[0][0]['info']['Qx']

            ''''''''''''''''''''''''''''''''''''''''''


            # Individual analysis

            # eval.infections_and_interventions_complete(save=True)
            eval.simulation_infection_plot(size_tup=(5.0, 3.7), granularity=0.001, save=True)
            # eval.simulation_treatment_plot(granularity=0.001, save=True)
            # eval.present_discounted_loss(plot=True, save=True)




            ''''''''''''''''''''''''''''''''''''''''''
            
            # Comparison analysis 

            # summary_tup = eval.infections_and_interventions_complete(size_tup = (8, 5), save=True)
            # multi_summary['infections_and_interventions'][saved[selected][0]] = summary_tup

            # summary_tup = eval.summarize_interventions_and_intensities()
            # multi_summary['stats_intervention_intensities'][saved[selected][0]] = summary_tup


            ''''''''''''''''''''''''''''''''''''''''''
            
            # eval.debug()

        # dum = (saved, all_selected, multi_summary)
        # joblib.dump(dum, 'multi_comp_dump_{}'.format(saved[all_selected[-1]][0]))

    else:

        dum = joblib.load('multi_comp_dump_{}'.format(saved[all_selected[-1]][0]))
        saved = dum[0]
        all_selected = dum[1]
        multi_summary = dum[2]

    '''Comparative analysis'''
    multi_eval = MultipleEvaluations(saved, all_selected, multi_summary)

    multi_eval.compare_infections(size_tup=(5.0, 3.7), save=True)
