
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
import maxcut


'''

Class that implements the simulation of the disease control dynamical system 

'''

class SISDynamicalSystem:

    def __init__(self, X_init, A, param, cost):

        self.N = A.shape[0]
        self.beta = param['beta']
        self.gamma = param['gamma']
        self.delta = param['delta']
        self.rho = param['rho']
        self.eta = param['eta']

        # LRSR
        self.spectral_ranking = None

        # MCM
        self.mcm_ranking = None

        # CURE
        self.is_waiting = True
        self.is_path_following_phase = False
        self.G = nx.from_numpy_matrix(A, parallel_edges=False, create_using=None)

        if len(X_init) == self.N and \
                A.shape[0] == A.shape[1]:

            self.A = A
            self.X_init = X_init
            self.Z_init = np.dot(A.T, X_init)
            self.Qlam = cost['Qlam'] * np.ones(self.N)
            self.Qx = cost['Qx'] * np.ones(self.N)

        else:
            print("\n--Dimensions don't match.--\n")
            exit(1)


    '''
    Simulates the dynamical system over the time period
    policy_fun must be of 'shape': (1, )        ->  (N, )
                           where   t in [0, T]  ->  control vector for all N nodes 
    '''

    def __simulate(self, policy_fun, time, plot, plot_update=1.0):

        # time initialization
        self.ttotal = time

        # state variable initialization
        self.X = np.array(
            [StochasticProcess(initial_condition=self.X_init[i]) for i in range(self.N)])
        self.Z = np.array(
            [StochasticProcess(initial_condition=self.Z_init[i]) for i in range(self.N)])
        self.H = np.array([StochasticProcess(initial_condition=0.0)
                           for _ in range(self.N)])
        self.M = np.array([StochasticProcess(initial_condition=0.0)
                           for _ in range(self.N)])
        self.Y = np.array([CountingProcess() for _ in range(self.N)])
        self.W = np.array([CountingProcess() for _ in range(self.N)])
        self.Nc = np.array([CountingProcess() for _ in range(self.N)])
        self.u = np.array([StochasticProcess(initial_condition=0.0)
                           for _ in range(self.N)])

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
                assert(not np.any([self.X[i].value_at(t)
                                   for i in range(self.N)]))
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
            p[p == 0.] = 0.  # sets -0.0 to 0.0
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
    Returns intensities of counting processes Y, W, N as defined by model
    '''

    def __getPoissonIntensities(self, t, policy_fun):
        '''Update control policy for every node'''
        for i, control in enumerate(policy_fun(t)):
            self.u[i].generate_arrival_at(t, None, N=control)

        '''Compute intensities according to model'''
        lambdaY = np.array([(1 - self.X[i].value_at(t)) * (self.beta * self.Z[i].value_at(
            t) - self.gamma * self.M[i].value_at(t)) for i in range(self.N)])
        lambdaW = np.array([self.X[i].value_at(
            t) * (self.delta + self.rho * self.H[i].value_at(t)) for i in range(self.N)])
        lambdaN = np.array([self.u[i].value_at(
            t) * self.X[i].value_at(t) * (1 - self.H[i].value_at(t)) for i in range(self.N)])
        return lambdaY, lambdaW, lambdaN


    '''
    Simulates any given policy
    
    SOC    - Stochastic optimal control
    TR     - Trivial
    TR-FL  - Trivial (front-loaded)
    MN     - Most-neighbors
    MN-FL  - Most-neighbors (front-loaded)
    LN     - Least-neighbors 
    LN-FL  - Least-neighbors (front-loaded)
    LRSR   - Largest reduction in spectral radius policy
    MCM    - MaxCut Minimization strategy, see
             http://kalogeratos.com/psite/files/MyPapers/MCM_NetworksWorkshos_NIPS2015.pdf 
    
    '''

    def simulate_policy(self, policy, baselines_dict, sim_dict, plot=False):

        time = sim_dict['total_time']

        if policy == 'SOC':
            return self.__simulate(self.__getOptPolicy, time, plot)
        
        elif policy == 'TR':
            return self.__simulate(lambda t: self.__getTrivialPolicy(baselines_dict['TR'], t), time, plot)
        
        elif policy == 'TR-FL':
            return self.__simulate(lambda t: self.__getTrivialPolicyFrontLoaded(baselines_dict['TR'], baselines_dict['FL_info'], t), time, plot)
        
        elif policy == 'MN':
            return self.__simulate(lambda t: self.__getMNDegreeHeuristicPolicy(baselines_dict['MN'], t), time, plot)

        elif policy == 'MN-FL':
            return self.__simulate(lambda t: self.__getMNDegreeHeuristicFrontLoaded(baselines_dict['MN'], baselines_dict['FL_info'], t), time, plot)

        elif policy == 'LN':
            return self.__simulate(lambda t: self.__getLNDegreeHeuristicPolicy(baselines_dict['LN'], t), time, plot)

        elif policy == 'LN-FL':
            return self.__simulate(lambda t: self.__getLNDegreeHeuristicFrontLoaded(baselines_dict['LN'], baselines_dict['FL_info'], t), time, plot)
        
        elif policy == 'LRSR':
            return self.__simulate(lambda t: self.__getLRSRHeuristicPolicy(baselines_dict['LRSR'], t), time, plot)

        elif policy == 'MCM':
            return self.__simulate(lambda t: self.__getMCMPolicy(baselines_dict['MCM'], t), time, plot)

        else:
            print('Invalid policy code.')
            exit(1)
        
    
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


    ''' *** Policies ***'''

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

    def __getTrivialPolicy(self, const, t):
        '''Array of X[i]'s and H[i]'s at time t'''
        X = np.array([self.X[i].value_at(t) for i in range(self.N)])
        H = np.array([self.H[i].value_at(t) for i in range(self.N)])

        state = np.multiply(X, np.ones(self.N) - H)
        return np.multiply(const * np.ones(self.N), np.multiply(state, self.Qx / self.Qlam))


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
    Returns Largest reduction in spectral radius (LRSR) heuristic policy u at time t
    u = 1/rank where rank is priority order of LRSR (independent of X(t) for consistency)
    const adjusts total intensity to match interventions with SOC
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

            # compute ranking
            self.spectral_ranking = np.argsort(l)

            # assign intensities according to order
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

    
    def __getMCMPolicy(self, const, t):
        """
        Returns the adapted heuristic policy MaxCutMinimzation (MCM) `u` at
        time `t`. The method is adapted to fit the setup where treatment
        intensity `rho` is the equal for everyone, and the control is made on
        the rate of intervention, not the intensity of the treatment itself.
        """
        if t <= 0:
            # Find MCM priority order once and cache it
            self.mcm_ranking = maxcut.mcm(self.A)
        elif self.mcm_ranking is None:
            # The priority order should be cached. Raise an exception.
            raise Exception('MCM ranking not computed...'
                            'Something went wrong.')
        ramp = const / np.arange(self.N, 0, -1)
        u = np.ones(self.N)
        u[self.mcm_ranking] = ramp
        return u



    '''
    Returns front-loaded variation of policy u at time t
    Front-loading means: 
    '''

    def __frontloadPolicy(self, u, frontload_info, t):

        # make sure limit of interventions is not reached yet
        max_count = frontload_info['N']
        interv_ct = np.sum([self.Nc[i].value_at(t) for i in range(self.N)])
        if interv_ct > max_count:
            u = np.zeros(self.N)

        # scale proportionally s.t. max(u) = max(u_SOC)
        if np.sum(u) > 0:

            # maximum treatment intensity of SOC over all trials
            max_SOC = frontload_info['max_u']

            # maximum treatment intensity of u over all trials
            max_u = np.max(u)

            return max_SOC * u / max_u
        else:
            return u


    '''
    Returns trivial policy u at time t, front-loaded to spent interventions earlier
    intensity(v) ~ Qx * Qlam^-1 * X * (1 - H), at maximum intensity of SOC
    '''

    def __getTrivialPolicyFrontLoaded(self, const, frontload_info, t):

        # compute front-loaded variant
        u = self.__getTrivialPolicy(const, t)
        return self.__frontloadPolicy(u, frontload_info, t)


    '''
    Returns MN (most neighbors) degree heuristic policy u at time t

    Front-loaded: maximum total intensity of SOC over time 
    is used distributed across all nodes at all times t
    '''

    def __getMNDegreeHeuristicFrontLoaded(self, const, frontload_info, t):

        # compute front-loaded variant
        u = self.__getMNDegreeHeuristicPolicy(const, t)
        return self.__frontloadPolicy(u, frontload_info, t)


    '''
    Returns LN (least neighbors) degree heuristic policy u at time t
    Front-loaded
    '''

    def __getLNDegreeHeuristicFrontLoaded(self, const, frontload_info, t):

         # compute front-loaded variant
        u = self.__getLNDegreeHeuristicPolicy(const, t)
        return self.__frontloadPolicy(u, frontload_info, t)
