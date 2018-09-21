
import numpy as np 
from pprint import pprint
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import reduce
import os

from helpers import HelperFunc

'''

Class that analyzes results of dynamical system simulations

'''

class Evaluation:

    def __init__(self, data, filename, description):
        self.data = data
        self.filename = filename
        self.descr = description

        # create directory for plots
        directory = 'plots/' + self.filename[:-4]
        if not os.path.exists(directory):
            os.makedirs(directory)


    ''' *** Helper Functions *** '''

    '''Creates parameter description string for textbox'''

    def __getTextBoxString(self):
        dict = self.data[0][0]['info']
        s_beta = r'$\beta$: ' + str(dict['beta']) + ', '
        s_delta = r'$\delta$: ' + str(dict['delta']) + ', '
        s_rho = r'$\rho$: ' + str(dict['rho']) + ', '
        s_gamma = r'$\gamma$: ' + str(dict['gamma']) + ', '
        s_eta = r'$\eta$: ' + str(dict['eta'])
        s_Qlam = r'Q$_{\lambda}$: ' + \
            str(np.mean(dict['Qlam'])) + ', '
        s_Qx = 'Q$_{X}$: ' + str(np.mean(dict['Qx']))
        s_sims = '# of simulations: ' + str(len(self.data[0]))

        s = s_beta + s_gamma + s_delta + s_rho + s_eta + '\n' \
            + s_Qlam + s_Qx + '\n' \
            + s_sims
        return s

    
    '''Computes integral from 0 to T of e^(eta * t) * f_of_t * dt for a given trial
    assuming f_of_t is tuple returned by HelperFunc.step_sps_values_over_time'''

    def __integrateF(self, f_of_t, eta):
        t, f = f_of_t

        # compute integral by summing integrals of constant intervals given by f_of_t
        int = 0.0
        indices = [(i, i + 1) for i in [2 * j for j in range(round(len(t) / 2))]]
        for i, j in indices:
            const = f[i]
            a, b = t[i], t[j]
            if eta == 0.0:
                # int_a^b  const * dt = const * (b - a)
                int += const * (b - a)
            else:
                # int_a^b  exp(- eta * t) * const * dt = const / eta * (exp(- a * eta) - exp(- b * eta))
                int += const / eta * (np.exp(- a * eta) - np.exp(- b * eta))

        return int

    '''Computes integral from 0 to T of (Qx * X) dt for a given trial'''

    def __computeIntX(self, trial, custom_eta=None):
        hf = HelperFunc()
        if custom_eta is None:
            eta = trial['info']['eta']
        else:
            eta = custom_eta
        X_, Qx = trial['X'], trial['info']['Qx']
        t, X = hf.step_sps_values_over_time(X_, summed=False)

        if X:
            f_of_t = t, np.dot(Qx, np.array(X).T)
        else:
            f_of_t = t, 0

        return self.__integrateF(f_of_t, eta)

    '''Computes integral from 0 to T of (0.5 * Qlam * u^2) dt for a given trial'''

    def __computeIntLambda(self, trial, custom_eta=None):
        hf = HelperFunc()
        if custom_eta is None:
            eta = trial['info']['eta']
        else:
            eta = custom_eta
        u_, Qlam = trial['u'], trial['info']['Qlam']
        t, u = hf.step_sps_values_over_time(u_, summed=False)
        
        if u:
            f_of_t = t, 0.5 * np.dot(Qlam, np.square(np.array(u)).T)
        else:
            f_of_t = t, 0

        return self.__integrateF(f_of_t, eta)


    '''Computes integral from 0 to T of |H|_1 dt for a given trial'''

    def __computeIntH(self, trial, custom_eta=None):
        hf = HelperFunc()
        if custom_eta is None:
            eta = trial['info']['eta']
        else:
            eta = custom_eta
        H_ = trial['H']
        t, H = hf.step_sps_values_over_time(H_, summed=False)

        if H:
            f_of_t = t, np.dot(np.ones(trial['info']['N']), np.array(H).T)
        else:
            f_of_t = t, 0

        return self.__integrateF(f_of_t, eta)
   

    ''' *** Population statistics *** '''

    '''Plots PDV of total incurred cost (i.e. the infinite horizon loss function)'''

    def present_discounted_loss(self, plot=False, save=False):
        
        
        '''Compute integral for every heuristic '''
        print("Computing present discounted loss integral for every heuristic...")
        pdvs_by_heuristic = [[self.__computeIntX(trial) + self.__computeIntLambda(trial) 
                             for trial in tqdm(heuristic)] for heuristic in self.data]
        means, stddevs = [np.mean(pdvs) for pdvs in pdvs_by_heuristic], [np.std(pdvs) for pdvs in pdvs_by_heuristic]
        print("...done.")

        '''Plotting functionality'''
        if plot:
            fig = plt.figure(figsize=(10, 6), facecolor='white')
            ax = fig.add_subplot(111, frameon=False)

            x = np.arange(len(means))
            width = 0.2
            ax.bar(x + width / 2, means, yerr=stddevs,
                width=width, align='center', color='rgbkymc')
            ax.set_xticks(x + width / 2)
            ax.set_xlabel('Policies')
            ax.set_xticklabels(self.descr)
            ax.set_ylabel('Present discounted loss')

            # text box
            box = True
            if box:
                s = self.__getTextBoxString()
                _, upper = ax.get_ylim()
                plt.text(0.0, 0.8 * upper, s, size=12,
                        va="baseline", ha="left", multialignment="left",
                        bbox=dict(fc="none"))

            plt.title("Cumulative present discounted loss (=objective function) for all heuristics")
            if save:
                plt.savefig('plots/' + self.filename[:-4] + '/PDV_plot.png', format='png', frameon=False)
            else:
                plt.show()
        
        print("\nPresent discounted loss (Mean, StdDev) \n")
        for j in range(len(self.data)):
            print(self.descr[j] + ':\t' + str(round(means[j], 3)) + '\t' + str(round(stddevs[j], 3)) )
        return 0

    def present_discounted_loss_BOX(self, plot=False, save=False):
        '''Compute integral for every heuristic'''
        print("Computing present discounted loss integral for every heuristic...")
        pdvs_by_heuristic = [[self.__computeIntX(trial) + self.__computeIntLambda(trial)
                              for trial in tqdm(heuristic)] for heuristic in self.data]
        print("...done.")

        '''Plotting functionality'''
        if plot:
            fig = plt.figure(figsize=(10, 6), facecolor='white')
            ax = fig.add_subplot(111, frameon=False)

            # x = np.arange(len(means))
            ax.boxplot(pdvs_by_heuristic, whis=[5, 95])
            ax.set_xlabel('Policies')
            ax.set_xticklabels(self.descr)
            ax.set_ylabel('Present discounted loss')
            ax.set_ylim((0, np.max(pdvs_by_heuristic)))

            plt.title("Cumulative present discounted loss (=objective function) for all heuristics")

            if save:
                plt.savefig('plots/' + self.filename[:-4] + '/PDV_BOXplot.png', format='png', frameon=False)
            else:
                plt.show()

        return 0

    '''Plots TOTAL infection cost (Qx.X) per TOTAL time under treatment (1.H)'''

    def infection_cost_per_time_under_treatment(self, plot=False, save=False):

        '''Compute (total infection cost / total time under treatment) for every heuristic'''
        print("Computing total infection cost / total time under treatment) for every heuristic...")
        infection_cost_per_interventions_by_heuristic = [[self.__computeIntX(trial, custom_eta=0.0) / self.__computeIntH(trial, custom_eta=0.0)
                                                          for trial in heuristic] for heuristic in self.data]
        means, stddevs = [np.mean(rel_costs) for rel_costs in infection_cost_per_interventions_by_heuristic], \
                         [np.std(rel_costs) for rel_costs in infection_cost_per_interventions_by_heuristic]
        print("...done.")

        '''Plotting functionality'''
        if plot:
            fig = plt.figure(figsize=(10, 6), facecolor='white')
            ax = fig.add_subplot(111, frameon=False)
            x = np.arange(len(means))
            width = 0.2
            ax.bar(x + width / 2, means, yerr=stddevs,
                width=width, align='center', color='rgbkymc')
            ax.set_xticks(x + width / 2)
            ax.set_xlabel('Policies')
            ax.set_xticklabels(self.descr)
            ax.set_ylabel('Relative infection cost incurred per time under treatment')

            plt.title("Total infection cost Int(Qx.X) divided by total time under treatment Int(1.H) for all heuristics")

            if save:
                plt.savefig('plots/' + self.filename[:-4] + 
                    '/infection_cost_per_time_under_treatment_plot.png', format='png', frameon=False)
            else:
                plt.show()
        
        print("\nRelative infection cost incurred per time under treatment (Mean, StdDev) \n")
        for j in range(len(self.data)):
            print(self.descr[j] + ':\t' + str(round(means[j], 3)) + '\t' + str(round(stddevs[j], 3)))
        return 0
    
    '''Plots TOTAL infection cost (Qx.X) & TOTAL time under treatment (1.H)'''

    def infection_cost_AND_time_under_treatment(self, plot=False, save=False):
        '''Compute total infection cost and total time under treatment for every heuristic'''
        print("Computing total infection cost and total time under treatment  for every heuristic...")
        infection_cost_by_heuristic = [[self.__computeIntX(trial, custom_eta=0.0)  for trial in heuristic] for heuristic in self.data]
        treatment_time_by_heuristic = [[self.__computeIntH(trial, custom_eta=0.0)  for trial in heuristic] for heuristic in self.data]
        print("...done.")

        means_infection, stddevs_infection = [np.mean(infections) for infections in infection_cost_by_heuristic], \
            [np.std(infections) for infections in infection_cost_by_heuristic]
        means_treatment, stddevs_treatment = [np.mean(treatments) for treatments in treatment_time_by_heuristic], \
            [np.std(treatments) for treatments in treatment_time_by_heuristic]

        '''Plotting functionality'''
        if plot:
            fig = plt.figure(figsize=(10, 6), facecolor='white')
            ax = fig.add_subplot(111, frameon=False)
            x = np.arange(len(means_infection))
            width = 0.2
            ax.bar(x, means_infection, yerr=stddevs_infection,
                   width=width, align='center', color='rgbkymc')
            ax.set_xticks(x + width / 2)
            ax.set_xlabel('Policies')
            ax.set_xticklabels(self.descr)
            ax.set_ylabel('Infection cost incurred [Left]')


            ax2 = ax.twinx()
            ax2.patch.set_visible(False)
            ax2.bar(x + width, means_treatment, yerr=stddevs_treatment,
                    width=width, align='center', color='rgbkymc', alpha=0.5)
            ax2.set_ylabel('Time under treatment [Right]')

            plt.title("Total infection cost Int(Qx.X) [Left] & Total time under treatment Int(1.H) [Right] for all heuristics")

            if save:
                plt.savefig('plots/' + self.filename[:-4] 
                            + '/infection_cost_AND_time_under_treatment.png', format='png', frameon=False)
            else:
                plt.show()
    
        print(
            "\nTotal infection cost and total time under treatment (Mean, StdDev) \n")
        for j in range(len(self.data)):
            print(self.descr[j] + ':\t' + str(round(means_infection[j], 3)) + '\t' + str(round(stddevs_infection[j], 3))
                    + '\t ---- \t' + str(round(means_treatment[j], 3)) + '\t' + str(round(stddevs_treatment[j], 3)))
        return 0
    
    '''Plots TOTAL infection cost (Qx.X) & TOTAL number of interventions (Sum of Nc[T])'''

    def infection_cost_AND_total_interventions(self, plot=False, save=False):
        '''Compute total infection cost and total number of interventions for every heuristic'''
        print("Computing total infection cost and total number of interventions for every heuristic...")
        hf = HelperFunc()
        infection_cost_by_heuristic = [[self.__computeIntX(trial, custom_eta=0.0) for trial in heuristic] for heuristic in self.data]
        # treatments_by_heuristic = [[np.sum(trial['Nc'][-1, :]) for trial in heuristic] for heuristic in self.data]
        treatments_by_heuristic = [[hf.sps_values(trial['Nc'], trial['info']['ttotal'], summed=True) 
                                    for trial in heuristic] for heuristic in self.data]
        print("...done.")
        
        means_infection, stddevs_infection = [np.mean(infections) for infections in infection_cost_by_heuristic], \
            [np.std(infections) for infections in infection_cost_by_heuristic]
        means_treatment, stddevs_treatment = [np.mean(treatments) for treatments in treatments_by_heuristic], \
            [np.std(treatments) for treatments in treatments_by_heuristic]

        '''Plotting functionality'''
        if plot:
            fig = plt.figure(figsize=(10, 6), facecolor='white')
            ax = fig.add_subplot(111, frameon=False)
            x = np.arange(len(means_infection))
            width = 0.2
            ax.bar(x, means_infection, yerr=stddevs_infection,
                   width=width, align='center', color='rgbkymc')
            ax.set_xticks(x + width / 2)
            ax.set_xlabel('Policies')
            ax.set_xticklabels(self.descr)
            ax.set_ylabel('Infection cost incurred [Left]')

            ax2 = ax.twinx()
            ax2.patch.set_visible(False)
            ax2.bar(x + width, means_treatment, yerr=stddevs_treatment,
                    width=width, align='center', color='rgbkymc', alpha=0.5)
            ax2.set_ylabel('Number of interventions [Right]')
           
            plt.title("Total infection cost Int(Qx.X) [Left] & Total number of interventions (Sum of N[Final time]) [Right] for all heuristics")

            if save:
                plt.savefig(
                    'plots/' + self.filename[:-4] 
                    + '/infection_cost_AND_total_interventions.png', format='png', frameon=False)
            else:
                plt.show()
        
        print(
            "\nTotal infection cost and total number of interventions (Mean, StdDev) \n")
        for j in range(len(self.data)):
            print(self.descr[j] + ':\t' + str(round(means_infection[j], 3)) + '\t' + str(round(stddevs_infection[j], 3))
                    + '\t ---- \t' + str(round(means_treatment[j], 3)) + '\t' + str(round(stddevs_treatment[j], 3)))
        return 0

    '''Plots TOTAL infection cost (Qx.X) & TOTAL intervention effort (Qlam.u^2)'''

    def infection_cost_AND_intervention_effort(self, plot=False, save=False):
        '''Compute total infection cost and total time under treatment for every heuristic'''
        print("Computing total infection cost and total time under treatment for every heuristic...")
        infection_cost_by_heuristic = [[self.__computeIntX(trial, custom_eta=0.0)  for trial in heuristic] for heuristic in self.data]
        treatment_time_by_heuristic = [[self.__computeIntLambda(trial, custom_eta=0.0)  for trial in heuristic] for heuristic in self.data]
        print("...done.")

        means_infection, stddevs_infection = [np.mean(infections) for infections in infection_cost_by_heuristic], \
            [np.std(infections) for infections in infection_cost_by_heuristic]
        means_treatment, stddevs_treatment = [np.mean(treatments) for treatments in treatment_time_by_heuristic], \
            [np.std(treatments) for treatments in treatment_time_by_heuristic]

        '''Plotting functionality'''
        if plot:
            fig = plt.figure(figsize=(10, 6), facecolor='white')
            ax = fig.add_subplot(111, frameon=False)
            x = np.arange(len(means_infection))
            width = 0.2
            ax.bar(x, means_infection, yerr=stddevs_infection,
                   width=width, align='center', color='rgbkymc')
            ax.set_xticks(x + width / 2)
            ax.set_xlabel('Policies')
            ax.set_xticklabels(self.descr)
            ax.set_ylabel('Infection cost incurred [Left]')


            ax2 = ax.twinx()
            ax2.patch.set_visible(False)
            ax2.bar(x + width, means_treatment, yerr=stddevs_treatment,
                    width=width, align='center', color='rgbkymc', alpha=0.5)
            ax2.set_ylabel('Treatment effort [Right]')


            plt.title( "Total infection cost Int(Qx.X) [Left] & Total intervention effort Int(Qlam.Lambda^2) [Right] for all heuristics")

            if save:
                plt.savefig(
                    'plots/' + self.filename[:-4] + 
                    '/infection_cost_AND_intervention_effort.png', format='png', frameon=False)
            else:
                plt.show()
        
        print(
            "\nTotal infection cost and total intervention effort (Mean, StdDev) \n")
        for j in range(len(self.data)):
            print(self.descr[j] + ':\t' + str(round(means_infection[j], 3)) + '\t' + str(round(stddevs_infection[j], 3))
                    + '\t ---- \t' + str(round(means_treatment[j], 3)) + '\t' + str(round(stddevs_treatment[j], 3)))
        return 0


    '''Simulation summary'''

    def simulation_infection_plot(self, granularity=0.001, save=False):

        print("Creating simulation infection plot...")
        
        '''Plotting functionality'''
        # Set up figure.
        fig = plt.figure(figsize=(12, 8), facecolor='white')
        ax = fig.add_subplot(111, frameon=False)

        plt.cla()

        hf = HelperFunc()

        colors = 'rgbkymc'
        for ind, heuristic in enumerate(self.data):

            tspace = np.arange(0.0, heuristic[0]['info']['ttotal'], granularity)
            values_in_tspace = np.array([[hf.sps_values(trial['X'], t, summed=True)
                                          for t in tspace] for trial in tqdm(heuristic)])
            mean_X_t = np.mean(values_in_tspace, axis=0)
            stddev_X_t = np.std(values_in_tspace, axis=0)
            
            ax.plot(tspace, mean_X_t, color=colors[ind])
            ax.fill_between(tspace, mean_X_t - stddev_X_t, mean_X_t + stddev_X_t,
                            alpha=0.3, edgecolor=colors[ind], facecolor=colors[ind],
                linewidth=0)

        ax.set_xlim([0, heuristic[0]['info']['ttotal']])
        ax.set_xlabel('Time')
        ax.set_ylim([0, heuristic[0]['info']['N']])
        ax.set_ylabel('Number of nodes')

        # text box
        box = True
        if box:
            s = self.__getTextBoxString()
            _, upper = ax.get_ylim()
            plt.text(0.0, upper, s, size=12,
                        va="baseline", ha="left", multialignment="left",
                        bbox=dict(fc="none"))

        legend = []
        for str in self.descr:
            legend += ['Total infected |X|: ' + str] #,
            #           str +  'Total under treatment |H|']
        ax.legend(legend)
        plt.draw()
        
        if save:
            plt.savefig(
                'plots/' + self.filename[:-4] + 
                '/simulation_infection_summary.png', format='png', frameon=False)
        else:
            plt.show()
        return 0

    def simulation_treatment_plot(self, granularity=0.001, save=False):
        
        print("Creating simulation treatment plot...")

        '''Plotting functionality'''

        # Set up figure.
        fig = plt.figure(figsize=(12, 8), facecolor='white')
        ax = fig.add_subplot(111, frameon=False)

        plt.cla()

        hf = HelperFunc()

        colors = 'rgbkymc'
        for ind, heuristic in enumerate(self.data):

            tspace = np.arange(0.0, heuristic[0]['info']['ttotal'], granularity)
            values_in_tspace = np.array([[hf.sps_values(trial['H'], t, summed=True)
                                          for t in tspace] for trial in tqdm(heuristic)])
            mean_H_t = np.mean(values_in_tspace, axis=0)
            stddev_H_t = np.std(values_in_tspace, axis=0)

            ax.plot(tspace, mean_H_t, color=colors[ind])
            ax.fill_between(tspace, mean_H_t - stddev_H_t, mean_H_t + stddev_H_t,
                            alpha=0.3, edgecolor=colors[ind], facecolor=colors[ind],
                            linewidth=0)

        ax.set_xlim([0, heuristic[0]['info']['ttotal']])
        ax.set_xlabel('Time')
        ax.set_ylim([0, heuristic[0]['info']['N']])
        ax.set_ylabel('Number of nodes')

        # text box
        box = True
        if box:
            s = self.__getTextBoxString()
            _, upper = ax.get_ylim()
            plt.text(0.0, upper, s, size=12,
                        va="baseline", ha="left", multialignment="left",
                        bbox=dict(fc="none"))

        legend = []
        for str in self.descr:
            legend += ['Total under treatment |H|: ' + str]
        ax.legend(legend)
        plt.draw()

        if save:
            plt.savefig(
                'plots/' + self.filename[:-4] + 
                '/simulation_treatment_summary.png', format='png', frameon=False)
        else:
            plt.show()

        return 0
