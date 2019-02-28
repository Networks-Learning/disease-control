import matplotlib.pyplot as plt
import collections
import joblib
import os

from analysis import Evaluation, MultipleEvaluations

plt.switch_backend('agg')

if __name__ == '__main__':


    '''Evaluate results'''

    saved = {
        0: 'test_all_but_MCM_Q_1_400_v5.pkl',
    }


    all_selected = [0]  # select pickle files to import
    multi_summary_from_dump = False


    # summary for multi setting comparison
    multi_summary = collections.defaultdict(dict)
    if not multi_summary_from_dump:

        '''Individual analyses'''
        for selected in all_selected:

            # to see graphs instead of saving them, comment out 'plt.switch_backend('agg')' from top of main.py
            # do not comment out when running on cluster

            print('Analyzing:  {}'.format(saved[selected]))

            filename = saved[selected]
            expname = os.path.splitext(filename)[0]
            data = joblib.load(os.path.join('temp_pickles', filename))
            description = [d['name'] for d in data]
            dat = [d['dat'] for d in data]
            evaluation = Evaluation(dat, expname, description)

            multi_summary['Qs'][saved[selected]] = evaluation.data[0][0]['info']['Qx'][0]

            ''''''''''''''''''''''''''''''''''''''''''

            '''Individual analysis'''

            evaluation.simulation_plot(
                process='X', filename='simulation_infection_summary',
                granularity=1.0, save=True)
            evaluation.simulation_plot(
                process='H', filename='simulation_treatment_summary',
                granularity=1.0, save=True)
            
            # evaluation.infections_and_interventions_complete(save=True)
            # evaluation.present_discounted_loss(plot=True, save=True)

            ''''''''''''''''''''''''''''''''''''''''''
            
            '''Compute Comparison analysis data'''

            # summary_tup = eval.infections_and_interventions_complete(size_tup = (8, 5), save=True)
            # multi_summary['infections_and_interventions'][saved[selected]] = summary_tup

            summary_tup = evaluation.summarize_interventions_and_intensities()
            # multi_summary['stats_intervention_intensities'][saved[selected]] = summary_tup


            ''''''''''''''''''''''''''''''''''''''''''
            # eval.debug()


        # dum = (saved, all_selected, multi_summary)
        # joblib.dump(dum, 'multi_comp_dump_{}'.format(saved[all_selected[-1]]))

    else:

        dum = joblib.load('multi_comp_dump_{}'.format(saved[all_selected[-1]]))
        saved = dum[0]
        all_selected = dum[1]
        multi_summary = dum[2]

    '''Comparative analysis'''
    # multi_eval = MultipleEvaluations(saved, all_selected[-1], multi_summary)

    # multi_eval.compare_infections(size_tup=(5.0, 3.7), save=True)
