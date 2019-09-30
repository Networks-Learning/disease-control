
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm, tqdm_notebook
import os

from helpers import HelperFunc

matplotlib.rcParams.update({
    "figure.autolayout": False,
    "figure.figsize": (8, 6),
    "figure.dpi": 72,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "xtick.minor.width": 0.8,
    "ytick.major.width": 0.8,
    "ytick.minor.width": 0.8,
    "text.usetex": True,
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "legend.fontsize": 14,
    "legend.frameon": True,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "lines.linewidth": 2.0,
    "lines.markersize": 4,
    "grid.linewidth": 0.4,
})


class Evaluation:
    """
    Class that analyzes results of dynamical system simulations
    """

    def __init__(self, data, plot_dir, description):
        self.data = data
        self.dirname = plot_dir
        self.descr = description

        self.colors = 'rggbbkkym'
        self.linestyles = ['-', '-', ':', '-', ':', '-', ':', '-', '-']

        # Create directory for plots
        if not os.path.exists(self.dirname):
            os.makedirs(self.dirname)

    ''' *** Helper Functions *** '''

    def __getTextBoxString(self):
        """
        Create parameter description string for textbox.
        """
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

    def __integrateF(self, f_of_t, eta):
        """
        Compute the integral from 0 to T of e^(eta * t) * f_of_t * dt for a
        given trial assuming f_of_t is tuple returned by
        `HelperFunc.step_sps_values_over_time`.
        """
        t, f = f_of_t

        # compute the integral by summing integrals of constant intervals
        # given by f_of_t
        val = 0.0
        indices = [(i, i + 1) for i in [2 * j for j in range(round(len(t) / 2))]]
        for i, j in indices:
            const = f[i]
            a, b = t[i], t[j]
            if eta == 0.0:
                # int_a^b  const * dt = const * (b - a)
                val += const * (b - a)
            else:
                # int_a^b  exp(- eta * t) * const * dt = const / eta * (exp(- a * eta) - exp(- b * eta))
                val += const / eta * (np.exp(- a * eta) - np.exp(- b * eta))
        return val

    def computeIntX(self, trial, custom_eta=None, weight_by_Qx=True):
        """
        Compute the integral from 0 to T of (Qx * X) dt for a given trial.
        """
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

    def __computeIntLambda(self, trial, custom_eta=None):
        """
        Compute integral from 0 to T of (0.5 * Qlam * u^2) dt 
        for a given trial.
        """
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

    def _computeIntH(self, trial, custom_eta=None):
        """
        Compute integral from 0 to T of |H|_1 dt for a given trial.
        """
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

    def present_discounted_loss(self, plot=False, save=False):
        """
        Plot PDV of total incurred cost
        (i.e. the infinite horizon loss function).
        """
        # Compute integral for every heuristic
        print(("Computing present discounted loss integral "
               "for every heuristic..."))
        pdvs_by_heuristic = [[self.computeIntX(trial) + self.__computeIntLambda(trial)
                             for trial in tqdm(heuristic)] for heuristic in self.data]
        means, stddevs = [np.mean(pdvs) for pdvs in pdvs_by_heuristic], [np.std(pdvs) for pdvs in pdvs_by_heuristic]
        print("...done.")

        # Plotting functionality
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

            plt.title(("Cumulative present discounted loss "
                       "(=objective function) for all heuristics"))
            if save:
                plt.savefig(os.path.join(self.dirname, 'PDV_plot.png'),
                            format='png', frameon=False)
                plt.close()
            else:
                plt.show()
        print("\nPresent discounted loss (Mean, StdDev) \n")
        for j in range(len(self.data)):
            print(self.descr[j] + ':\t' + str(round(means[j], 3)) + '\t' + str(round(stddevs[j], 3)) )
        return 0

    def infections_and_interventions_complete(self, size_tup=(15, 10), save=False):
        """
        Summarizes simulations in 3 plots
        - Infection coverage (Int X(t) dt) - Total discrete interventions (Sum N(T))
        - Infection coverage (Int X(t) dt) - Treatment coverage (Int H(t) dt)
        - Infection events   (Sum Y(T))    - Total discrete interventions (Sum N(T))
        """

        # Compute statistics for every heuristic
        hf = HelperFunc()
        intX_by_heuristic = [[self.computeIntX(trial, custom_eta=0.0, weight_by_Qx=False)
                              for trial in heuristic] for heuristic in tqdm_notebook(self.data)]

        intX_m = np.array([np.mean(h) for h in intX_by_heuristic])
        intX_s = np.array([np.std(h) for h in intX_by_heuristic])

        intH_by_heuristic = [[self._computeIntH(trial, custom_eta=0.0)
                              for trial in heuristic] for heuristic in tqdm_notebook(self.data)]

        intH_m = np.array([np.mean(h) for h in intH_by_heuristic])
        intH_s = np.array([np.std(h) for h in intH_by_heuristic])

        Y_by_heuristic = [[hf.sps_values(trial['Y'], trial['info']['ttotal'], summed=True)
                           for trial in heuristic] for heuristic in tqdm_notebook(self.data)]

        Y_m = np.array([np.mean(h) for h in Y_by_heuristic])
        Y_s = np.array([np.std(h) for h in Y_by_heuristic])

        N_by_heuristic = [[hf.sps_values(trial['Nc'], trial['info']['ttotal'], summed=True)
                           for trial in heuristic] for heuristic in tqdm_notebook(self.data)]

        N_m = np.array([np.mean(h) for h in N_by_heuristic])
        N_s = np.array([np.std(h) for h in N_by_heuristic])

        x = np.arange(len(intX_m))
        n = 50  # trials per simulation

        # Plotting functionality
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        fig = plt.figure(figsize=(12, 8), facecolor='white')

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
        plt.tight_layout()

        if save:
            fig_filename = os.path.join(self.dirname, "infections_and_interventions_complete" + '.pdf')
            plt.savefig(fig_filename, format='pdf', frameon=False, dpi=300)
            plt.close()
        else:
            plt.show()

        return ((intX_m, intX_s), (N_m, N_s), (intH_m, intH_s), (Y_m, Y_s))

    def infection_cost_AND_intervention_effort(self, plot=False, save=False):
        """
        Plots the TOTAL infection cost (Qx.X) and the TOTAL intervention
        effort (Qlam.u^2).
        """
        # Compute total infection cost and total time under treatment
        # for every heuristic
        print(("Computing total infection cost and total time under treatment "
               "for every heuristic..."))
        infection_cost_by_heuristic = [[self.computeIntX(trial, custom_eta=0.0)  for trial in heuristic] for heuristic in self.data]
        treatment_time_by_heuristic = [[self.__computeIntLambda(trial, custom_eta=0.0)  for trial in heuristic] for heuristic in self.data]
        print("...done.")

        means_infection, stddevs_infection = [np.mean(infections) for infections in infection_cost_by_heuristic], \
            [np.std(infections) for infections in infection_cost_by_heuristic]
        means_treatment, stddevs_treatment = [np.mean(treatments) for treatments in treatment_time_by_heuristic], \
            [np.std(treatments) for treatments in treatment_time_by_heuristic]

        # Plotting functionality
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

            plt.title(("Total infection cost Int(Qx.X) [Left] & "
                       "Total intervention effort Int(Qlam.Lambda^2) "
                       "[Right] for all heuristics"))

            if save:
                plt.savefig(os.path.join(
                    self.dirname, 'infection_cost_AND_intervention_effort.png'),
                    format='png', frameon=False)
                plt.close()
            else:
                plt.show()

        print(("\nTotal infection cost and total intervention effort "
               "(Mean, StdDev) \n"))
        for j in range(len(self.data)):
            print(self.descr[j] + ':\t' + str(round(means_infection[j], 3)) +
                  '\t' + str(round(stddevs_infection[j], 3)) +
                  '\t ---- \t' + str(round(means_treatment[j], 3)) +
                  '\t' + str(round(stddevs_treatment[j], 3)))
        return 0

    def summarize_interventions_and_intensities(self):
        """
        Return total number of interventions & peak and average treatment
        intensities for every heuristic.
        """

        hf = HelperFunc()

        # Intensities
        max_intensities = np.zeros((len(self.data), len(self.data[0])), dtype=object)
        for i, heuristic in enumerate(tqdm(self.data)):
            for j, trial in enumerate(heuristic):
                all_arrivals = hf.all_arrivals(trial['u'])
                max_intensities[i, j] = np.zeros(len(all_arrivals))
                for k, t in enumerate(all_arrivals):
                    max_intensities[i, j][k] = np.max(
                        hf.sps_values(trial['u'], t, summed=False))

        max_per_trial = [[np.max(trial) for trial in heuristic]
                         for heuristic in tqdm(max_intensities)]
        max_per_heuristic = [np.max(heuristic) for heuristic in max_per_trial]

        print(max_per_trial)
        print(max_per_heuristic)

        # Treatments

        # treatments_by_heuristic = [[hf.sps_values(trial['Nc'], trial['info']['ttotal'], summed=True)
        #                             for trial in heuristic] for heuristic in self.data]

        # means_treatment, stddevs_treatment = \
        #     [np.mean(treatments) for treatments in treatments_by_heuristic], \
        #     [np.std(treatments) for treatments in treatments_by_heuristic]

        return 0  # TODO Change back and delete this

    def simulation_plot(self, process, figsize=(8, 6), granularity=0.1,
                        filename='simulation_summary', draw_box=False,
                        save=False):
        """
        Plot a summary of the simulation.

        Parameters
        ----------
        process : str
            Process to plot (`X`, `H`, `Y`, `W`, `Nc`, `u`)
        figsize : tuple, optional
            Figure size
        granularity : float, optional
            Time (x-axis) granularity of the plot
        filename : str, optional
            Filename to save the plot
        save : bool, optional
            If True, the plot is saved to `filename`
        draw_box : bool, optional
            If True, draw a text box with simulation parameters
        """

        print(f"Building simulation figure for process {process:s}...")

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        hf = HelperFunc()

        for i, heuristic in enumerate(self.data):
            # Number of trials
            n_trials = len(heuristic)
            # Simulation end time
            ttotal = heuristic[0]['info']['ttotal']
            # Linearly spaced array of time
            tspace = np.arange(0.0, ttotal, granularity)
            # Extract the values of the stochastic processes at all times
            # for each trial
            values = np.zeros((n_trials, len(tspace)))
            for j, trial in enumerate(tqdm_notebook(heuristic)):
                for k, t in enumerate(tspace):
                    values[j, k] = hf.sps_values(trial[process], t, summed=True)
            # Compute mean and std over trials
            mean_X_t = np.mean(values, axis=0)
            stddev_X_t = np.std(values, axis=0)
            # Plot the mean +/- std
            ax.plot(tspace, mean_X_t, color=self.colors[i], linestyle=self.linestyles[i])
            ax.fill_between(tspace, mean_X_t - stddev_X_t, mean_X_t + stddev_X_t,
                            alpha=0.3, edgecolor=self.colors[i], facecolor=self.colors[i], 
                            linewidth=0)
        ax.set_xlim([0, ttotal])
        ax.set_xlabel("Elapsed time")
        # ax.set_ylim([0, heuristic[0]['info']['N']])
        ax.set_ylim(bottom=0)
        if process == 'X':
            ax.set_ylabel("Number of infected nodes")
        elif process == 'H':
            ax.set_ylabel("Number of treated nodes")
        else:
            ax.set_ylabel("Number of nodes")

        # Text box
        if draw_box:
            s = self.__getTextBoxString()
            _, upper = ax.get_ylim()
            plt.text(0.0, upper, s, size=12,
                     va="baseline", ha="left", multialignment="left",
                     bbox=dict(fc="none"))
        # Legend
        legend = []
        for policy in self.descr:
            legend += [policy]
        ax.legend(legend)
        plt.draw()

        if save:
            fig_filename = os.path.join(self.dirname, filename + '.pdf')
            plt.savefig(fig_filename, format='pdf', frameon=False, dpi=300)
            plt.close()
        else:
            plt.show()


class MultipleEvaluations:
    """
    Class that plots results of multiple dynamical system simulations.
    """

    def __init__(self, multi_summary, policy_list, n_trials, save_dir):
        """

        Arguments:
        ----------
        multi_summary : dict
            Data for the plot formatted as follows:
            {
                'Qs': {
                    'expname_1': `qs_1`, (`qs_1` is a float corresponding for parameter qs)
                    'expname_2': `qs_2`,
                    ...
                },
                'infections_and_interventions': {
                    'expname_1': (
                        (
                            `intX_m`, (mean of integral of process X over the observation period)
                            `intX_s` (std of integral of process X over the observation period)
                        ),
                        (
                            `N_m`, (mean of process N over the observation period)
                            `N_s` (std of process N over the observation period)
                        ),
                        (
                            `intH_m`, (mean of integral of process H over the observation period)
                             `intH_s` (std of integral of process H over the observation period)
                        ),
                        (
                            `Y_m`, (mean of process Y over the observation period)
                            `Y_s` (mean of process Y over the observation period)
                        )
                    ),
                    ...
                    ()

                }
            }
        """
        self.multi_summary = multi_summary
        self.policy_list = policy_list
        self.n_trials = n_trials

        self.colors = 'rggbbkkym'
        self.linestyles = ['-', '-', ':', '-', ':', '-', ':', '-', '-']

        # create directory for plots
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def compare_infections(self, size_tup=(8, 6), save=True):
        """
        Compare infections along Qx axis (assuming Qlambda = const = 1.0).
        """
        d = self.multi_summary.get('infections_and_interventions', None)
        Qs = self.multi_summary.get('Qs', None)

        if d is None or Qs is None:
            raise ValueError('Missing data.')

        keys = np.array(list(Qs.keys()))
        n_exps = len(keys)

        # assumes data had same methods tested
        infections_axis = {name: np.zeros(n_exps) for name in self.policy_list}
        infections_axis_std = {name: np.zeros(n_exps) for name in self.policy_list}
        interventions_axis = {name: np.zeros(n_exps) for name in self.policy_list}
        interventions_axis_std = {name: np.zeros(n_exps) for name in self.policy_list}
        
        # Build X-axis
        Qx_axis = np.array([Qs[key] for key in keys])
        # Sort by value
        sorted_args = np.argsort(Qx_axis)
        Qx_axis = Qx_axis[sorted_args]
        keys = keys[sorted_args]

        # Build Y-axis
        for i, name in enumerate(self.policy_list):
            for j, key in enumerate(keys):
                # d[key] = ((intX_m, intX_s), (N_m, N_s), (intH_m, intH_s), (Y_m, Y_s))
                infections_axis[name][j] = d[key][0][0][i]
                infections_axis_std[name][j] = d[key][0][1][i] / np.sqrt(self.n_trials)  # transfrom stddev into std error
                interventions_axis[name][j] = d[key][1][0][i]
                interventions_axis_std[name][j] = d[key][1][1][i] / np.sqrt(self.n_trials)  # transfrom stddev into std error



        # Set up figure.
        fig, ax = plt.subplots(1, 1, figsize=(12, 8), facecolor='white')
        plt.cla()

        hf = HelperFunc()

        legend = []
        max_infected = 0
        for ind, name in enumerate(self.policy_list):
            legend.append(name)

            # linear axis
            ax.plot(Qx_axis, infections_axis[name], color=self.colors[ind],
                    linestyle=self.linestyles[ind])
            ax.fill_between(
                Qx_axis,
                infections_axis[name] - interventions_axis_std[name],
                infections_axis[name] + interventions_axis_std[name],
                alpha=0.3,
                edgecolor=self.colors[ind],
                facecolor=self.colors[ind],
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
            plt.savefig(os.path.join(self.save_dir, 'infections_fair_comparison.png'), frameon=False, format='png', dpi=dpi)
            plt.close()
        else:
            plt.show()
