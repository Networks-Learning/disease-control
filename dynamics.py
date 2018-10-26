
import pandas as pd
import numpy as np 
from pprint import pprint
import matplotlib.pyplot as plt
import scipy.stats
from tqdm import tqdm
import scipy.optimize
from time import sleep
from functools import reduce
import networkx as nx

from stochastic_processes import StochasticProcess, CountingProcess
from helpers import HelperFunc 


'''

Class that implements the simulation of the disease control dynamical system 

'''

class SISDynamicalSystem:

    def __init__(self, N, X_init, A, param, cost):

        self.N = N
        self.beta = param['beta']
        self.gamma = param['gamma']
        self.delta = param['delta']
        self.rho = param['rho']
        self.eta = param['eta']

        # LRSR
        self.spectral_ranking = None

        # CURE
        self.is_waiting = True
        self.is_path_following_phase = False
        self.G = nx.from_numpy_matrix(A, parallel_edges=False, create_using=None)

        if len(X_init) == N and \
                A.shape[0] == N and \
                A.shape[1] == N and \
                len(cost['Qlam']) == N and \
                len(cost['Qx']) == N:

            self.A = A
            self.X_init = X_init
            self.Z_init = np.dot(A.T, X_init)
            self.Qlam = cost['Qlam']
            self.Qx = cost['Qx']

        else:
            print("\n--Dimensions don't match.--\n")
            exit(1)

    
    '''
    Simulates the dynamical system using the stochastic optimal control policy
    '''

    def simulate_opt(self, time, plot=False, plot_update=1.0):
        return self.__simulate(self.__getOptPolicy, time, plot, plot_update)

    '''
    Simulates the dynamical system using a trivial control intensity
    '''

    def simulate_trivial(self, intensity_const, time, plot=False, plot_update=1.0):
        return self.__simulate(lambda t: self.__getTrivialPolicy(intensity_const, t), time, plot, plot_update)

    '''
    Simulates the dynamical system using a trivial control intensity, 
    but front-loaded with a max on number of interventions, then zero after the max is reached
    '''

    def simulate_trivial_frontloaded(self, level, frontld_dict, const, time, plot=False, plot_update=1.0):
        return self.__simulate(lambda t: self.__getTrivialPolicyFrontLoaded(level, frontld_dict, const, t), time, plot, plot_update)

    '''
    Simulates the dynamical system using a trivial control intensity, 
    using the same total intensity as OPT but redistributed over nodes
    '''

    def simulate_trivial_online_comparison(self, time, plot=False, plot_update=1.0):
        return self.__simulate(lambda t: self.__getTrivialPolicyOnlineComparison(t), time, plot, plot_update)



    '''
    Simulates the dynamical system using the MN (most neighbors) degree heuristic
    '''

    def simulate_MN_degree_heuristic(self, const, time, plot=False, plot_update=1.0):
        return self.__simulate(lambda t: self.__getMNDegreeHeuristicPolicy(const, t), time, plot, plot_update)


    '''
    Simulates the dynamical system using the MN (most neighbors) degree heuristic
    but front-loaded with a max on number of interventions, then zero after the max is reached
    '''

    def simulate_MN_frontloaded(self, level, frontld_dict, const, time, plot=False, plot_update=1.0):
        return self.__simulate(lambda t: self.__getMNDegreeHeuristicFrontLoaded(level, frontld_dict, const, t), time, plot, plot_update)


    '''
    Simulates the dynamical system using the MN (most neighbors) degree heuristic
    using the same total intensity as OPT but redistributed over nodes
    '''

    def simulate_MN_degree_online_comparison(self, time, plot=False, plot_update=1.0):
        return self.__simulate(lambda t: self.__getMNDegreeHeuristicPolicyOnlineComparison(t), time, plot, plot_update)



    '''
    Simulates the dynamical system using the LN (least neighbors) degree heuristic
    '''

    def simulate_LN_degree_heuristic(self, const, time, plot=False, plot_update=1.0):
        return self.__simulate(lambda t: self.__getLNDegreeHeuristicPolicy(const, t), time, plot, plot_update)

    '''
    Simulates the dynamical system using the LN (least neighbors) degree heuristic
    but front-loaded with a max on number of interventions, then zero after the max is reached
    '''

    def simulate_LN_frontloaded(self, level, frontld_dict, const, time, plot=False, plot_update=1.0):
        return self.__simulate(lambda t: self.__getLNDegreeHeuristicFrontLoaded(level, frontld_dict, const, t), time, plot, plot_update)




    '''
    Simulates the dynamical system using the Largest reduction in spectral radius(LRSR) heuristic 
    '''

    def simulate_LRSR_heuristic(self, const, time, plot=False, plot_update=1.0):
        return self.__simulate(lambda t: self.__getLRSRHeuristicPolicy(const, t), time, plot, plot_update)

    '''
    Returns CURE  policy u at time t, adjusted to fit our model
    Implemented from https://arxiv.org/pdf/1407.2241.pdf 
    Instead of investing all resources in one node, we can only focus all 
    intensity on one node as this is the control signal
    '''

    def simulate_CURE_policy(self, frontld_dict, const, time, plot=False, plot_update=1.0):
        return self.__simulate(lambda t: self.__getCUREPolicy(frontld_dict, const, t), time, plot, plot_update)



    '''
    Simulates the dynamical system using the CUT/priority order heuristic 
    from http://kalogeratos.com/psite/files/MyPapers/MCM_NetworksWorkshos_NIPS2015.pdf
    with adjustments to fit the realistic setup where treatment improvement rho is the same for everyone
    and as such we control the _intensity_ of intervening, not the intensity of the treatment itself
    '''

    def simulate_MCM(self, frontld_dict, resource_const, time, plot=False, plot_update=1.0):
        return self.__simulate(lambda t: self.__getMCMPolicy(frontld_dict, resource_const, t), time, plot, plot_update)
    

    '''
    Simulates the dynamical system over the time period
    policy_fun must be of 'shape': (1, )        ->  (N, )
                           where   t in [0, T]  ->  control vector for all N nodes 
    '''

    def __simulate(self, policy_fun, time, plot, plot_update):

        # time initialization
        self.ttotal = time['total']

        # state variable initialization
        self.X = np.array([StochasticProcess(initial_condition=self.X_init[i]) for i in range(self.N)])
        self.Z = np.array([StochasticProcess(initial_condition=self.Z_init[i]) for i in range(self.N)])
        self.H = np.array([StochasticProcess(initial_condition=0.0) for _ in range(self.N)])
        self.M = np.array([StochasticProcess(initial_condition=0.0) for _ in range(self.N)])
        self.Y = np.array([CountingProcess() for _ in range(self.N)])
        self.W = np.array([CountingProcess() for _ in range(self.N)])
        self.Nc = np.array([CountingProcess() for _ in range(self.N)])
        self.u = np.array([StochasticProcess(initial_condition=0.0) for _ in range(self.N)])

        # create infection events for initial infections
        for i in range(len(self.X_init)):
            if self.X_init[i] == 1.0:
                self.Y[i].generate_arrival_at(0.0)

        '''Simulate path over time'''
        t = 0.0
        self.all_processes = CountingProcess()
        self.last_arrival_type = None
        progress_bar = tqdm(total=self.ttotal, leave=False)

        
        '''Plotting functionality'''
        if plot:
            # Set up figure.
            fig = plt.figure(figsize=(12, 8), facecolor='white')
            ax = fig.add_subplot(111, frameon=False)
            # plt.ion()
            # plt.show(block=False)
            self.already_plotted = set()

            
            def callback(t, final=False):
                new_plot = divmod(t, plot_update)[0]
                if new_plot not in self.already_plotted or final:

                    self.already_plotted.add(new_plot)

                    plt.cla()

                    # string creation
                    s_beta = r'$\beta$: ' + str(self.beta) + ', '
                    s_delta = r'$\delta$: ' + str(self.delta) + ', '
                    s_rho = r'$\rho$: ' + str(self.rho) + ', '
                    s_gamma = r'$\gamma$: ' + str(self.gamma) + ', '
                    s_eta = r'$\eta$: ' + str(self.eta)
                    s_Qlam = r'Q$_{\lambda}$: ' + \
                        str(np.mean(self.Qlam)) + ', '
                    s_Qx = 'Q$_{X}$: ' + str(np.mean(self.Qx))
                    s = s_beta + s_gamma + '\n' \
                        + s_delta + s_rho + s_eta + '\n' \
                        + s_Qlam + s_Qx

                    # plotting
                    plt.text(0.0, self.N, s, size=12,
                             va="baseline", ha="left", multialignment="left",
                             bbox=dict(fc="none"))

                    # helper functions
                    hf = HelperFunc()

                    tX, yX = hf.step_sps_values_over_time(self.X, summed=True)
                    tH, yH = hf.step_sps_values_over_time(self.H, summed=True)

                    print(tX, yX)
                    ax.plot(tX, yX)
                    ax.plot(tH, yH)

                    ax.set_xlim([0, self.ttotal])
                    ax.set_xlabel('Time')
                    ax.set_ylim([0, self.N])
                    ax.set_ylabel('Number of nodes')

                    ax.legend(['Total infected |X|', 'Total under treatment |H|'])
                    plt.draw()
                    plt.pause(1e-10)


        while t < self.ttotal:
            
            
            '''Compute sum of intensities'''
            lambdaY, lambdaW, lambdaN = self.__getPoissonIntensities(t, policy_fun)
            lambda_all = np.sum(lambdaY) + np.sum(lambdaW) + np.sum(lambdaN)
            if lambda_all == 0:
                # infection went extinct
                # print("Infection went extinct at t = {}".format(round(t, 3)))
                assert(not np.any([self.X[i].value_at(t) for i in range(self.N)]))
                progress_bar.update(self.ttotal - t)
                t = self.ttotal
                break
            
            '''Generate next arrival of all processes Y[i], W[i], N[i]'''
            u = np.random.uniform(0.0, 1.0)
            w = - np.log(u) / lambda_all
            t = t + w
            self.all_processes.generate_arrival_at(t)
            progress_bar.update(w)

            

            '''Sample what process the arrival came from'''
            p = np.hstack((lambdaY, lambdaW, lambdaN)) / lambda_all
            p[p == 0.] = 0. # sets -0.0 to 0.0
            k = np.random.choice(3 * self.N, p=p)


            '''Adjust state variables accordingly'''
            # arrival Y
            if k < self.N:
                self.last_arrival_type = 'Y'
                i = k 
                self.Y[i].generate_arrival_at(t)

                # dX = dY - dW
                self.X[i].generate_arrival_at(t, 1.0)

                # dZ = A(dY - dW)
                for j in np.where(self.A[i])[0]:
                    self.Z[j].generate_arrival_at(t, 1.0)

            # arrival W
            elif self.N <= k < 2 * self.N:
                self.last_arrival_type = 'W'
                i = k - self.N
                self.W[i].generate_arrival_at(t)

                # dX = dY - dW
                self.X[i].generate_arrival_at(t, -1.0)

                # dZ = A(dY - dW)
                for j in np.where(self.A[i])[0]:
                    self.Z[j].generate_arrival_at(t, -1.0)

                # dH = dN - H.dW
                prev_H = self.H[i].value_at(t)
                if prev_H == 1:
                    self.H[i].generate_arrival_at(t, -1.0)
                
                # dM = A(dN - H.dW)
                if prev_H == 1:
                    for j in np.where(self.A[i])[0]:
                        self.M[j].generate_arrival_at(t, -1.0)

            # arrival N
            else:
                self.last_arrival_type = 'N'
                i = k - 2 * self.N
                self.Nc[i].generate_arrival_at(t)

                # dH = dN - H.dW
                self.H[i].generate_arrival_at(t, 1.0)

                # dM = A(dN - H.dW)
                for j in np.where(self.A[i])[0]:
                    self.M[j].generate_arrival_at(t, 1.0)

            if plot:
                callback(t)
                #pass
            

        if plot:
            callback(t, final=True)
            plt.ioff()
            plt.show()

        progress_bar.close()
        
        # return collected data for analysis
        return self.__getCollectedData()
    
    '''
    Creates file containing all information of the past simulation for analysis 
    '''
    def __getCollectedData(self):
        info = {'N': self.N,
                'beta': self.beta,
                'delta': self.delta,
                'rho': self.rho,
                'gamma': self.gamma,
                'eta': self.eta,
                'ttotal': self.ttotal,
                'Qlam' : self.Qlam,
                'Qx': self.Qx}

        return {'info': info,
                'X': self.X,
                'H': self.H,
                'Y': self.Y,
                'W': self.W,
                'Nc': self.Nc,
                'u': self.u}



    '''
    Returns intensities of counting processes Y, W, N as defined by model
    '''

    def __getPoissonIntensities(self, t, policy_fun):
        '''Update control policy for every node'''
        for i, control in enumerate(policy_fun(t)):
            self.u[i].generate_arrival_at(t, None, N=control)
        
        '''Compute intensities according to model'''
        lambdaY = np.array([(1 - self.X[i].value_at(t)) * (self.beta * self.Z[i].value_at(t) - self.gamma * self.M[i].value_at(t)) for i in range(self.N)])
        lambdaW = np.array([self.X[i].value_at(t) * (self.delta + self.rho * self.H[i].value_at(t)) for i in range(self.N)])
        lambdaN = np.array([self.u[i].value_at(t) * self.X[i].value_at(t) * (1 - self.H[i].value_at(t)) for i in range(self.N)])
        return lambdaY, lambdaW, lambdaN

    '''
    Returns stochastic optimal control policy u at time t
    '''

    def __getOptPolicy(self, t):
        
        '''Array of X[i]'s and H[i]'s at time t'''
        X = np.array([self.X[i].value_at(t) for i in range(self.N)])
        H = np.array([self.H[i].value_at(t) for i in range(self.N)])

        '''Only recompute if X changed at last arrival'''
        if self.last_arrival_type is None or self.last_arrival_type == 'Y' or self.last_arrival_type == 'W':
            

            '''Convert X into array of indices where X = 0 (indices_X_is_0) and X = 1 (indices_X_is_1)'''
            indices_X_is_0 = np.where(np.ones(self.N) - X)[0]
            indices_X_is_1 = np.where(X)[0]
            cnt_X_is_0 = indices_X_is_0.shape[0]
            cnt_X_is_1 = indices_X_is_1.shape[0]

            Qx_1 = np.delete(self.Qx, indices_X_is_0)
            Qlam_1 = np.delete(self.Qlam, indices_X_is_0)
            A_10 = np.delete(np.delete(self.A, indices_X_is_1, axis=1),
                            indices_X_is_0, axis=0)

            '''Delta_V^N = (1 / K1) * [K2 - sqrt(2*K1*Qlam * (K3 * A_10.d_0 + K4) + K2^2)] (see writeup)'''
            K1 = self.beta * (2 * self.delta + self.eta + self.rho)
            K2 = self.beta * (self.delta + self.eta) * \
                (self.delta + self.eta + self.rho) * Qlam_1
            K3 = self.eta * (self.gamma * (self. delta + self.eta) +
                            self.beta * (self.delta + self.rho))
            K4 = self.beta * (self.delta + self.rho) * Qx_1

            '''LP: find d_0 s.t. slack in inequality is minimized. See appendix of writeup on how to formulate LP.'''
            obj = np.hstack((np.ones(cnt_X_is_1), np.zeros(cnt_X_is_0)))
            epsilon = 0  # 1e-11 # to make sqrt definitely pos.; scipy.optimize.linprog tolerance is 1e-12
            epsilon_expr = epsilon / (2 * K1 * Qlam_1 * K3)
            A_ineq = np.hstack((np.zeros((cnt_X_is_1, cnt_X_is_1)), - A_10))
            b_ineq = K4 / K3 - epsilon_expr
            A_eq = np.hstack((np.eye(cnt_X_is_1), - A_10))
            b_eq = K4 / K3 - epsilon_expr

            result = scipy.optimize.linprog(obj, A_ub=A_ineq, b_ub=b_ineq, A_eq=A_eq, b_eq=b_eq)
            if result['success']:
                d_0 = result['x'][cnt_X_is_1:]
            else:
                print(result)
                print("LP couldn't be solved.")
                exit(1)

            Delta_N = (1 / K1) * (K2 - np.sqrt(2 * K1 * Qlam_1 *
                                               (K3 * np.dot(A_10, d_0) + K4) + np.square(K2)))
            Delta_N_full = np.zeros(self.N)
            Delta_N_full[indices_X_is_1] = Delta_N

            '''Check invariant ensured by LP: Delta_V^N is non-positive.'''
            try:
                assert(not np.any(K3 * np.dot(A_10, d_0) + K4 < 0))
            except AssertionError:
                print(
                    "AssertionError: LP didn't manage to make sqrt() definitely greater than K2. \n \
                    Consider setting epsilon > 0 in code. ")
                exit(1)

            self.last_opt = np.multiply(np.multiply(- Delta_N_full, 1.0 / self.Qlam),
                                        np.multiply(X, np.ones(self.N) - H))
        

        return self.last_opt
        

    '''
    Returns trivial policy u at time t
    intensity(v) ~ Qx * Qlam^-1 * X * (1 - H)
    '''

    def __getTrivialPolicy(self, intensity_const, t):
        '''Array of X[i]'s and H[i]'s at time t'''
        X = np.array([self.X[i].value_at(t) for i in range(self.N)])
        H = np.array([self.H[i].value_at(t) for i in range(self.N)])

        
        state = np.multiply(X, np.ones(self.N) - H)
        return np.multiply(intensity_const * np.ones(self.N), np.multiply(state, self.Qx / self.Qlam))

    '''
    Returns trivial policy u at time t, front-loaded to spent interventions earlier
    intensity(v) ~ Qx * Qlam^-1 * X * (1 - H)

    - interpolates intensity between trivial_const and OPT_peak
    - level x in [1, 2, 3] treates 1/4, 1/2, 1 intensity of OPT_peak + trivial_const
      level 0 would correspond to constant trivial policy
    '''

    def __getTrivialPolicyFrontLoaded(self, level, frontld_dict, trivial_const, t):

        max_count = frontld_dict['N']
        peak_opt = frontld_dict['peak_OPT']

        '''Array of X[i]'s and H[i]'s at time t'''
        X = np.array([self.X[i].value_at(t) for i in range(self.N)])
        H = np.array([self.H[i].value_at(t) for i in range(self.N)])
        state = np.multiply(X, np.ones(self.N) - H)

        # make sure limit of interventions is not reached yet
        interv_ct = np.sum([self.Nc[i].value_at(t) for i in range(self.N)])
        if interv_ct > max_count:
            conditional_ones = np.zeros(self.N)
        else:
            conditional_ones = np.ones(self.N)

        # compute interpolated intensity for front-loading
        trivial_intensity = trivial_const * np.mean(self.Qx) / np.mean(self.Qlam)
        diff = peak_opt / self.N - trivial_intensity
        control = trivial_intensity + 1.0 / (2.0 ** (3 - level)) * diff

        return np.multiply(control * conditional_ones, state)



    '''
    Returns trivial policy u at time t, using the same total treatment intensity as OPT 
    at a given state.
    This is equivalent to redistributing the mass of intensities to a different pattern.

    '''

    def __getTrivialPolicyOnlineComparison(self, t):
        '''Array of X[i]'s and H[i]'s at time t'''
        X = np.array([self.X[i].value_at(t) for i in range(self.N)])
        H = np.array([self.H[i].value_at(t) for i in range(self.N)])
        state = np.multiply(X, np.ones(self.N) - H)

        opt_u = self.__getOptPolicy(t)
        opt_u_effective = np.multiply(opt_u, state)
        total = np.sum(opt_u_effective)
        m = np.sum(state)
        control = total / m if m > 0 else 0

        return np.multiply(np.multiply(control, np.ones(self.N)), state)



    '''
    Returns MN (most neighbors) degree heuristic policy u at time t
    intensity(v) ~ deg(v) * Qx * Qlam^-1 * X * (1 - H)
    '''

    def __getMNDegreeHeuristicPolicy(self, const, t):
        '''Array of X[i]'s and H[i]'s at time t'''
        X = np.array([self.X[i].value_at(t) for i in range(self.N)])
        H = np.array([self.H[i].value_at(t) for i in range(self.N)])


        deg = np.dot(self.A.T, np.ones(self.N))
        state = np.multiply(X, np.ones(self.N) - H)
        return np.multiply(const * deg, np.multiply(state, self.Qx / self.Qlam))


    '''
    Returns MN (most neighbors) degree heuristic policy u at time t
    intensity(v) ~ deg(v) * Qx * Qlam^-1 * X * (1 - H)

    - interpolates intensity between trivial_const and OPT_peak
    - level x in [1, 2, 3] treates 1/4, 1/2, 1 intensity of OPT_peak + trivial_const
    - level 0 would correspond to constant trivial policy
    '''

    def __getMNDegreeHeuristicFrontLoaded(self, level, frontld_dict, const, t):

        max_count = frontld_dict['N']
        peak_opt = frontld_dict['peak_OPT']

        '''Array of X[i]'s and H[i]'s at time t'''
        X = np.array([self.X[i].value_at(t) for i in range(self.N)])
        H = np.array([self.H[i].value_at(t) for i in range(self.N)])
        state = np.multiply(X, np.ones(self.N) - H)

        # make sure limit of interventions is not reached yet
        interv_ct = np.sum([self.Nc[i].value_at(t) for i in range(self.N)])
        if interv_ct > max_count:
            conditional_ones = np.zeros(self.N)
        else:
            conditional_ones = np.ones(self.N)

        # compute interpolated intensity for front-loading
        deg = np.dot(self.A.T, np.ones(self.N))
        deg_intensity = const * deg * np.mean(self.Qx) / np.mean(self.Qlam)
        diff = (peak_opt / self.N) * np.ones(self.N) - deg_intensity
        control = deg_intensity + 1.0 / (2.0 ** (3 - level)) * diff

        return np.multiply(np.multiply(control, conditional_ones), state)





    '''
    Returns MN policy u at time t, using the same total treatment intensity as OPT 
    at a given state.
    This is equivalent to redistributing the mass of intensities to a different pattern.

    '''

    def __getMNDegreeHeuristicPolicyOnlineComparison(self, t):

        '''Array of X[i]'s and H[i]'s at time t'''
        X = np.array([self.X[i].value_at(t) for i in range(self.N)])
        H = np.array([self.H[i].value_at(t) for i in range(self.N)])
        state = np.multiply(X, np.ones(self.N) - H)

        deg = np.dot(self.A.T, np.ones(self.N))
        total_degs_effective = np.sum(np.multiply(deg, state))

        opt_u = self.__getOptPolicy(t)
        opt_u_effective = np.multiply(opt_u, state)
        total = np.sum(opt_u_effective)

        control = total / total_degs_effective if total_degs_effective > 0 else 0

        return np.multiply(control * deg, state)

    '''
    Returns LN (least neighbors) degree heuristic policy u at time t
    intensity(v) ~ (maxdeg - deg(v) + 1) * Qx * Qlam^-1 * X * (1 - H)
    '''

    def __getLNDegreeHeuristicPolicy(self, const, t):
        '''Array of X[i]'s and H[i]'s at time t'''
        X = np.array([self.X[i].value_at(t) for i in range(self.N)])
        H = np.array([self.H[i].value_at(t) for i in range(self.N)])


        deg = np.dot(self.A.T, np.ones(self.N))
        state = np.multiply(X, np.ones(self.N) - H)
        return np.multiply(const * ((np.max(deg) + 1) * np.ones(self.N) - deg), np.multiply(state, self.Qx / self.Qlam))

    '''
    Returns LN (least neighbors) degree heuristic policy u at time t
    intensity(v) ~ (maxdeg - deg(v) + 1) * Qx * Qlam^-1 * X * (1 - H)
    but front-loaded
    '''

    def __getLNDegreeHeuristicFrontLoaded(self, level, frontld_dict, const, t):

        max_count = frontld_dict['N']
        peak_opt = frontld_dict['peak_OPT']

        '''Array of X[i]'s and H[i]'s at time t'''
        X = np.array([self.X[i].value_at(t) for i in range(self.N)])
        H = np.array([self.H[i].value_at(t) for i in range(self.N)])
        state = np.multiply(X, np.ones(self.N) - H)

        # make sure limit of interventions is not reached yet
        interv_ct = np.sum([self.Nc[i].value_at(t) for i in range(self.N)])
        if interv_ct > max_count:
            conditional_ones = np.zeros(self.N)
        else:
            conditional_ones = np.ones(self.N)

        # compute interpolated intensity for front-loading
        deg = np.dot(self.A.T, np.ones(self.N))
        deg_intensity = const * \
            ((np.max(deg) + 1) * np.ones(self.N) - deg) * \
            np.mean(self.Qx) / np.mean(self.Qlam)
        diff = (peak_opt / self.N) * np.ones(self.N) - deg_intensity
        control = deg_intensity + 1.0 / (2.0 ** (3 - level)) * diff

        return np.multiply(np.multiply(control, conditional_ones), state)




    '''
    Returns Largest reduction in spectral radius (LRSR) heuristic policy u at time t
    u = 1/rank where rank is priority order of LRSR (independent of X(t) for consistency)
    const adjusts total intensity to match interventions with OPT
    '''
    
    def __getLRSRHeuristicPolicy(self, const, t):

        def spectral_radius(A):
                return np.max(np.abs(np.linalg.eigvals(A)))

        if t <= 0:
            
            # Brute force: find which node removals reduce spectral radius the most
            tau = spectral_radius(self.A)

            l = np.zeros(self.N)
            for n in range(self.N):
                A_ = np.copy(self.A)
                A_[n, :] = np.zeros(self.N)
                A_[:, n] = np.zeros(self.N)
                l[n] = tau - spectral_radius(A_)

            self.spectral_ranking = np.argsort(l)
            ramp = const * np.flip(np.fromfunction(
                lambda i: 1 / (i + 1), (self.N, ), dtype=float))
            u = np.ones(self.N)
            u[self.spectral_ranking] = ramp

        else:
            if self.spectral_ranking is not None:
                ramp = const * np.flip(np.fromfunction(lambda i: 1/(i + 1), (self.N, ), dtype=int))
                u = np.ones(self.N)
                u[self.spectral_ranking] = ramp

            else:
                print('Spectral radius not computed. Something went wrong.')
                exit(1)

        return u


    '''Takes G, bag A, bag B, with A subseteq B
       returns crusade (w_0, w_1, ..., w_k) (sequence of bags)
          w_0 = A
          w_k = 0
          one node removed every step
       optimal means width of crusade is minimized
    '''

    def find_opt_crusade(self, G, bag_A):

        lookup_table = dict()

        def impedance(bag, crusade):
                        
            if len(bag) == 1:
                # recursion bottoms out
                return nx.cut_size(G, bag), [bag]
            else:
                # memoization: check if bag was already seen
                t = tuple(bag)
                if t in lookup_table:
                    return lookup_table[t][0], lookup_table[t][1]

                else:
                    all_impedances = []
                    all_crusades = []
                    
                    for i in range(len(bag)):
                        bag_minus_i = np.delete(bag, i).tolist()
                        im, cr = impedance(bag_minus_i, crusade + [bag_minus_i])
                        all_impedances.append(im)
                        all_crusades.append(cr)

                    argmin = np.argmin(all_impedances)

                    lookup_table[t] = np.max([nx.cut_size(G, bag), all_impedances[argmin]]), \
                                      [bag] + all_crusades[argmin]

                    return np.max([nx.cut_size(G, bag), all_impedances[argmin]]), [bag] + all_crusades[argmin]

        return impedance(bag_A, [bag_A])


    '''
    Returns CURE  policy u at time t, adjusted to fit our model
    Implemented from https://arxiv.org/pdf/1407.2241.pdf 
    Instead of investing all resources in one node, we can only focus all 
    intensity on one node as this is the control signal
    '''
    

    def __getCUREPolicy(self, frontld_dict, const, t):

        max_count = frontld_dict['N']
        peak_opt = 100.0 # ave_u # frontld_dict['peak_OPT']

        peak_opt = frontld_dict['peak_OPT']

        def get_I_t(sps, t_):
            return np.where(np.array([sps[i].value_at(t_) for i in range(self.N)]))[0].tolist()


        I_t = get_I_t(self.X, t)
        r = peak_opt

        
        if t <= 0:
            self.bag_A = I_t
            self.bag_B = None
            self.bag_C = None
            self.bag_D = None
            self.target_path = None

        # check invariants of impedance/optimal crusade
        check_invariants_impedance = False
        if check_invariants_impedance:
            impedance, opt_crus = self.find_opt_crusade(self.G, I_t)

            # delta(w_i+1) <= delta(w_) for i = 0, 1, . . . , k − 1.
            for k in range(len(opt_crus) - 1):
                w_k = self.find_opt_crusade(self.G, opt_crus[k])[0]
                w_k_1 = self.find_opt_crusade(self.G, opt_crus[k + 1])[0]
                print(k, w_k, w_k_1, w_k_1 <= w_k)
            
            # c(A) <= delta(A)
            print(nx.cut_size(self.G, I_t.tolist()), impedance, 
                  nx.cut_size(self.G, I_t.tolist()) <= impedance)


        if self.is_waiting:
            
            print('Waiting... Infected:', len(I_t))

            '''Waiting period'''
            cut = nx.cut_size(self.G, I_t)

            if cut <= r / 8:
                self.is_waiting = False
                self.bag_B = I_t

                print('Finding optimal crusade...')
                impedance, opt_crus = self.find_opt_crusade(self.G, self.bag_B)
                self.target_path = opt_crus

                print(opt_crus)
                exit(1)

        else:

            '''Segments'''
            if self.is_path_following_phase:
                
                '''Path-following phase'''
            
            else:

                '''Excursion'''




        return np.zeros(self.N)



    '''
    Returns cut heuristic policy u at time t
    Implemented from http://kalogeratos.com/psite/files/MyPapers/MCM_NetworksWorkshos_NIPS2015.pdf
    with adjustments to fit the realistic setup where treatment improvement rho is the same for everyone
    and as such we control the _intensity_ of intervening, not the intensity of the treatment itself
    '''

    def __getMCMPolicy(self, frontld_dict, resource_const, t):

        max_count = frontld_dict['N']
        peak_opt = frontld_dict['peak_OPT']


        '''Find priority order'''
        
        '''Define resource threshold r as r = c * N where N is the size of the network
           In the experiements of the paper, c was 0.1 and 0.25,  or 1.00 and 2.00
           In particular, they choose r = beta * maxcut'''  
        # r = resource_const * self.N
        
        # TODO



        print("TODO")

        exit(1)


        return np.ones(self.N)
