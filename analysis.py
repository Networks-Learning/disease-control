
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

        self.colors = 'rggbbkkym'
        self.linestyles = ['-', '-', ':', '-', ':', '-', ':', '-', '-']

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
        s_sims = 'no. of simulations: ' + str(len(self.data[0]))

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

    def computeIntX(self, trial, custom_eta=None, weight_by_Qx=True):
        hf = HelperFunc()
        if custom_eta is None:
            eta = trial['info']['eta']
        else:
            eta = custom_eta
        X_, Qx = trial['X'], trial['info']['Qx']
        t, X = hf.step_sps_values_over_time(X_, summed=False)

        if X:
            if weight_by_Qx:
                f_of_t = t, np.dot(Qx, np.array(X).T)
            else:
                f_of_t = t, np.dot(np.ones(Qx.shape), np.array(X).T)
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
   

    ''' *** ANALYSIS *** '''

    '''Plots PDV of total incurred cost (i.e. the infinite horizon loss function)'''

    def present_discounted_loss(self, plot=False, save=False):
        
        
        '''Compute integral for every heuristic '''
        print("Computing present discounted loss integral for every heuristic...")
        pdvs_by_heuristic = [[self.computeIntX(trial) + self.__computeIntLambda(trial) 
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
                width=width, align='center', color='rgbkymcgbkymc')
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


    '''
    Summarizes simulations in 3 plots
        - Infection coverage (Int X(t) dt) - Total discrete interventions (Sum N(T))  
        - Infection coverage (Int X(t) dt) - Treatment coverage (Int H(t) dt)     
        - Infection events   (Sum Y(T))    - Total discrete interventions (Sum N(T))  
    
    '''
    def infections_and_interventions_complete(self, size_tup = (15, 10), save=False):
        

        '''Compute statistics for every heuristic'''
        hf = HelperFunc()
        intX_by_heuristic = [[self.computeIntX(trial, custom_eta=0.0, weight_by_Qx=False) 
                              for trial in heuristic] for heuristic in tqdm(self.data)]

        intX_m = np.array([np.mean(h) for h in intX_by_heuristic])
        intX_s = np.array([np.std(h) for h in intX_by_heuristic])

        intH_by_heuristic = [[self.__computeIntH(trial, custom_eta=0.0)
                              for trial in heuristic] for heuristic in tqdm(self.data)]

        intH_m = np.array([np.mean(h) for h in intH_by_heuristic])
        intH_s = np.array([np.std(h) for h in intH_by_heuristic])


        Y_by_heuristic = [[hf.sps_values(trial['Y'], trial['info']['ttotal'], summed=True) 
                           for trial in heuristic] for heuristic in tqdm(self.data)]
        
        Y_m = np.array([np.mean(h) for h in Y_by_heuristic])
        Y_s = np.array([np.std(h) for h in Y_by_heuristic])

        N_by_heuristic = [[hf.sps_values(trial['Nc'], trial['info']['ttotal'], summed=True) 
                           for trial in heuristic] for heuristic in tqdm(self.data)]
                            
        N_m = np.array([np.mean(h) for h in N_by_heuristic])
        N_s = np.array([np.std(h) for h in N_by_heuristic])

        x = np.arange(len(intX_m))
        n = 50 # trials per simulation

        '''Plotting functionality'''
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        fig = plt.figure(figsize=(12, 8), facecolor='white')
        # fig = plt.figure()

        # 1 - Infection coverage (Int X(t) dt) - Total discrete interventions (Sum N(T))
        ax = fig.add_subplot(2, 1, 1, frameon=False)
        
        width = 0.2
        ax.bar(x, intX_m, yerr=intX_s / np.sqrt(n),
                width=width, align='center', color='rgbkymcgbkymc')
        ax.set_xticks(x + width / 2)
        ax.set_xlabel(r'Policies')
        ax.set_xticklabels(self.descr)
        ax.set_ylabel(r'Infection coverage $\int_{t_0}^{t_f} \mathbf{X}(t) dt$')

        ax2 = ax.twinx()
        ax2.patch.set_visible(False)
        ax2.bar(x + width, N_m, yerr=N_s / np.sqrt(n),
                width=width, align='center', color='rgbkymcgbkymc', alpha=0.5)
        ax2.set_ylabel(r'Interventions $\sum_{i=1}^{|nodes|} \mathbf{N}_i(t_f)$')
        
        plt.title(r"Infection coverage and discrete interventions")

        # 2 - Infection coverage (Int X(t) dt) - Treatment coverage (Int H(t) dt)
        ax = fig.add_subplot(2, 2, 3, frameon=False)
        
        width = 0.2
        ax.bar(x, intX_m, yerr=intX_s / np.sqrt(n),
                width=width, align='center', color='rgbkymcgbkymc')
        ax.set_xticks(x + width / 2)
        ax.set_xlabel(r'Policies')
        ax.set_xticklabels(['']*len(self.descr))
        ax.set_ylabel(r'Infection coverage $\int_{t_0}^{t_f} \mathbf{X}(t) dt$')

        ax2 = ax.twinx()
        ax2.patch.set_visible(False)
        ax2.bar(x + width, intH_m, yerr=intH_s / np.sqrt(n),
                width=width, align='center', color='rgbkymcgbkymc', alpha=0.5)
        ax2.set_ylabel(r'Treatment coverage $\int_{t_0}^{t_f} \mathbf{H}(t) dt$')
        
        plt.title(r"Infection coverage and treatment coverage")

        # 3 - Infection events   (Sum Y(T))    - Total discrete interventions (Sum N(T))
        ax = fig.add_subplot(2, 2, 4, frameon=False)
        
        width = 0.2
        ax.bar(x, intX_m, yerr=intX_s / np.sqrt(n),
                width=width, align='center', color='rgbkymcgbkymc')
        ax.set_xticks(x + width / 2)
        ax.set_xlabel(r'Policies')
        ax.set_xticklabels(['']*len(self.descr))
        ax.set_ylabel(r'Infections $\sum_{i=1}^{|nodes|} \mathbf{Y}_i(t_f)$')

        ax2 = ax.twinx()
        ax2.patch.set_visible(False)
        ax2.bar(x + width, N_m, yerr=N_s / np.sqrt(n),
                width=width, align='center', color='rgbkymcgbkymc', alpha=0.5)
        ax2.set_ylabel(r'Interventions $\sum_{i=1}^{|nodes|} \mathbf{N}_i(t_f)$')
        
        plt.title(r"Infection events and discrete interventions")

        if save:
            dpi = 400
            plt.tight_layout()
            fig = plt.gcf()  # get current figure
            fig.set_size_inches(size_tup)  # width, height
            plt.savefig(
                'plots/' + self.filename[:-4] 
                + '/infections_and_interventions_complete.png', format='png', frameon=False, dpi=dpi)
        else:
            plt.show()
    
    
        return ((intX_m, intX_s),(N_m, N_s), (intH_m, intH_s), (Y_m, Y_s))

    '''Plots TOTAL infection cost (Qx.X) & TOTAL intervention effort (Qlam.u^2)'''

    def infection_cost_AND_intervention_effort(self, plot=False, save=False):
        '''Compute total infection cost and total time under treatment for every heuristic'''
        print("Computing total infection cost and total time under treatment for every heuristic...")
        infection_cost_by_heuristic = [[self.computeIntX(trial, custom_eta=0.0)  for trial in heuristic] for heuristic in self.data]
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
                   width=width, align='center', color='rgbkymcgbkymc')
            ax.set_xticks(x + width / 2)
            ax.set_xlabel('Policies')
            ax.set_xticklabels(self.descr)
            ax.set_ylabel('Infection cost incurred [Left]')


            ax2 = ax.twinx()
            ax2.patch.set_visible(False)
            ax2.bar(x + width, means_treatment, yerr=stddevs_treatment,
                    width=width, align='center', color='rgbkymcgbkymc', alpha=0.5)
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

    
    '''Return total number of interventions & peak and average treatment intensities for every heuristic'''

    def summarize_interventions_and_intensities(self):

        hf = HelperFunc()
        
        '''Intensities'''
        max_intensities = [[[np.max(hf.sps_values(trial['u'], t, summed=False)) for t in hf.all_arrivals(trial['u'])]
                            for trial in heuristic] for heuristic in tqdm(self.data)]
        max_per_trial = [[np.max(trial) for trial in heuristic]
                         for heuristic in tqdm(max_intensities)]
        max_per_heuristic = [np.max(heuristic) for heuristic in max_per_trial]

        print(max_per_trial)
        print(max_per_heuristic)

        '''Treatments'''

        # treatments_by_heuristic = [[hf.sps_values(trial['Nc'], trial['info']['ttotal'], summed=True)
        #                             for trial in heuristic] for heuristic in self.data]

        # means_treatment, stddevs_treatment = \
        #     [np.mean(treatments) for treatments in treatments_by_heuristic], \
        #     [np.std(treatments) for treatments in treatments_by_heuristic]

        return 0 # TODO Change back and delete this


    '''Simulation summary'''

    def simulation_infection_plot(self, size_tup=(4,3), granularity=0.001, save=False):

        print("Creating simulation infection plot...")
        
        '''Plotting functionality'''
        # Set up figure.
        plt.rc('text', usetex=True)

        fig = plt.figure(figsize=(12, 8), facecolor='white')
        ax = fig.add_subplot(111, frameon=False)

        plt.cla()

        hf = HelperFunc()

        for ind, heuristic in enumerate(self.data):

            tspace = np.arange(0.0, heuristic[0]['info']['ttotal'], granularity)
            values_in_tspace = np.array([[hf.sps_values(trial['X'], t, summed=True)
                                          for t in tspace] for trial in tqdm(heuristic)])
            mean_X_t = np.mean(values_in_tspace, axis=0)
            stddev_X_t = np.std(values_in_tspace, axis=0)
            
            ax.plot(tspace, mean_X_t, color=self.colors[ind], linestyle=self.linestyles[ind])
            ax.fill_between(tspace, mean_X_t - stddev_X_t, mean_X_t + stddev_X_t,
                            alpha=0.3, edgecolor=self.colors[ind], facecolor=self.colors[ind],
                linewidth=0)

        ax.set_xlim([0, heuristic[0]['info']['ttotal']])
        ax.set_xlabel(r'$t$')
        ax.set_ylim([0, heuristic[0]['info']['N']])
        ax.set_ylabel(r'Infected nodes $\mathbf{1}^\top \mathbf{X}(t)$')

        # text box
        box = False
        if box:
            s = self.__getTextBoxString()
            _, upper = ax.get_ylim()
            plt.text(0.0, upper, s, size=12,
                        va="baseline", ha="left", multialignment="left",
                        bbox=dict(fc="none"))

        legend = []
        for str in self.descr:
            legend += [str]
        ax.legend(legend)
        plt.draw()
        
        if save:
            dpi = 300
            # plt.tight_layout()
            fig = plt.gcf()  # get current figure
            fig.set_size_inches(size_tup)  # width, height
            plt.savefig(
                'plots/' + self.filename[:-4] +
                '/simulation_infection_summary_withleg_2.png', format='png', frameon=False, dpi=dpi)
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

        for ind, heuristic in enumerate(self.data):

            tspace = np.arange(0.0, heuristic[0]['info']['ttotal'], granularity)
            values_in_tspace = np.array([[hf.sps_values(trial['H'], t, summed=True)
                                          for t in tspace] for trial in tqdm(heuristic)])
            mean_H_t = np.mean(values_in_tspace, axis=0)
            stddev_H_t = np.std(values_in_tspace, axis=0)

            ax.plot(tspace, mean_H_t, color=self.colors[ind])
            ax.fill_between(tspace, mean_H_t - stddev_H_t, mean_H_t + stddev_H_t,
                            alpha=0.3, edgecolor=self.colors[ind], facecolor=self.colors[ind],
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



    '''Debugging playground'''

    def debug(self):

        print('Debugging..')

        hf = HelperFunc()
        # infections_by_heuristic1 = [[self.computeIntX(trial, custom_eta=0.0, weight_by_Qx=False) 
        #                             for trial in heuristic] for heuristic in self.data]
        # infections_by_heuristic2 = [[hf.sps_values(trial['Y'], trial['info']['ttotal'], summed=True)
        #                              for trial in heuristic] for heuristic in self.data]
        # treatments_by_heuristic = [[hf.sps_values(trial['Nc'], trial['info']['ttotal'], summed=True) 
        #                             for trial in heuristic] for heuristic in self.data]
        # print("...done.")
        
        # infections = np.array(infections_by_heuristic2)
        # treatments = np.array(treatments_by_heuristic)
        # diff = infections - treatments

        trial = self.data[0][2]
        infections = np.array(hf.sps_values(trial['Y'], trial['info']['ttotal'], summed=False))
        infected_at_0 = np.array(hf.sps_values(trial['X'], 0.0, summed=False))
        infections_at_0 = np.array(hf.sps_values(trial['Y'], 0.0, summed=False))
        treatments = np.array(hf.sps_values(trial['Nc'], trial['info']['ttotal'], summed=False))

        print(infections)
        print(infected_at_0)
        print(infections_at_0)
        print(treatments)

        diff = infections - treatments
        print(diff)






'''

Class that plots results of multiple dynamical system simulations

'''


class MultipleEvaluations:

    def __init__(self, saved, all_selected, multi_summary):

        self.saved = saved 
        self.all_selected = all_selected 
        self.multi_summary = multi_summary
        
        self.colors = 'rggbbkkym'
        self.linestyles = ['-', '-', ':', '-', ':', '-', ':', '-', '-']

        # create directory for plots
        directory = 'plots_multi/' + str(self.all_selected)
        if not os.path.exists(directory):
            os.makedirs(directory)

    '''Compares infections along Qx axis (assuming Qlambda = const = 1.0)'''

    def compare_infections(self, size_tup = (5,5),save=True):
        
        
        d = self.multi_summary.get('infections_and_interventions', None)
        Qs = self.multi_summary.get('Qs', None)

        if d is not None and Qs is not None:
            
            keys = [self.saved[selected] for selected in self.all_selected]
            descriptions = [self.saved[selected][1] for selected in self.all_selected]

            # assumes data had same methods tested
            Qx_axis = [Qs[key] for key in keys]
            infections_axis = {name : [] for name in descriptions[0]} 
            infections_axis_std = {name : [] for name in descriptions[0]} 

            interventions_axis = {name : [] for name in descriptions[0]} 
            interventions_axis_std = {name : [] for name in descriptions[0]} 

            # transfrom stddev into std error
            n = 30 # trials per simulation

            # gather data from inputs
            for i, name in enumerate(descriptions[0]):
                for key in keys:
                    # d[key] = ((intX_m, intX_s), (N_m, N_s), (intH_m, intH_s), (Y_m, Y_s))
                    infections_axis[name].append(d[key][0][0][i])
                    infections_axis_std[name].append(d[key][0][1][i])
                    interventions_axis[name].append(d[key][1][0][i])
                    interventions_axis_std[name].append(d[key][1][1][i])
            

            '''Plotting functionality'''

            plt.rc('text', usetex=True)
            # plt.rc('font', family='serif')

            # Set up figure.
            fig = plt.figure(figsize=(12, 8), facecolor='white')
            ax = fig.add_subplot(111, frameon=False)
            
            # ax.set_xscale("log", nonposx='clip')
            # plt.xscale('log')

            plt.cla()

            hf = HelperFunc()

            legend = []
            max_infected = 0
            for ind, name in enumerate(descriptions[0]):
                legend.append(name)

                # linear axis
                ax.plot(Qx_axis, infections_axis[name], color=self.colors[ind], linestyle=self.linestyles[ind])
                ax.fill_between(Qx_axis, 
                                np.array(infections_axis[name]) - np.array(interventions_axis_std[name]  / np.sqrt(n)), 
                                np.array(infections_axis[name]) + np.array(interventions_axis_std[name]  / np.sqrt(n)),
                                alpha=0.3, edgecolor=self.colors[ind], facecolor=self.colors[ind],
                                linewidth=0)

                # ax.errorbar(Qx_axis, infections_axis[name], yerr=interventions_axis_std[name])

                if max(infections_axis[name]) > max_infected:
                    max_infected = max(infections_axis[name])

            ax.set_xlim([0.7, max(Qx_axis)])
            ax.set_xlabel(r'$Q_x$')
            ax.set_ylim([0, 1.3 * max_infected])
            ax.set_ylabel(r'Infection coverage $\int_{t_0}^{t_f} \mathbf{X}(t) dt$')
            ax.legend(legend)

            # ax.set_xscale("log", nonposx='clip')


            if save:
                dpi = 300
                # plt.tight_layout()
                fig = plt.gcf()  # get current figure
                fig.set_size_inches(size_tup)  # width, height
                plt.savefig(
                    'plots_multi/' + str(self.all_selected) + \
                    '/infections_fair_comparison.png', frameon=False, format='png', dpi=dpi)  #
            else:
                plt.show()                

        else:
            print('Error: Missing data.')
            exit(1)



