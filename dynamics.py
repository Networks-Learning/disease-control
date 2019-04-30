
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.optimize
import networkx as nx

from stochastic_processes import StochasticProcess, CountingProcess
from helpers import HelperFunc
import maxcut


class SISDynamicalSystem:
    """
    Class that implements the simulation of the disease control dynamical
    system.
    """

    def __init__(self, X_init, A, param, cost, verbose=True):
        self.n_nodes = A.shape[0]  # Number of nodes
        self.beta = param['beta']  # Infection rate
        self.gamma = param['gamma']  # Reduc.in infection rate from treatment
        self.delta = param['delta']  # Recovery rate (spontaneous)
        self.rho = param['rho']  # Recovery rate from treatment
        self.eta = param['eta']  # Exponential discount rate for SOC strategy

        # LRSR
        self.spectral_ranking = None

        # MCM
        self.mcm_ranking = None

        # CURE
        self.is_waiting = True
        self.is_path_following_phase = False
        self.G = nx.from_numpy_matrix(A, parallel_edges=False,
                                      create_using=None)

        self.verbose = verbose

        if (len(X_init) == self.n_nodes) and (A.shape[0] == A.shape[1]):
            self.A = A
            self.X_init = X_init
            self.Z_init = np.dot(A.T, X_init)
            self.Qlam = cost['Qlam'] * np.ones(self.n_nodes)
            self.Qx = cost['Qx'] * np.ones(self.n_nodes)
        else:
            raise ValueError("Dimensions don't match")

    def _simulate(self, policy_fun, time, plot, plot_update=1.0):
        """
        Simulate the SIS dynamical system using Ogata's thinning algorithm over
        the time period policy_fun must be of shape: (1,) -> (N,)
        where t in [0, T] -> control vector for all N nodes.
        """

        # Initialize the end time of the simulation
        self.ttotal = time

        # Init the array of infection state processes
        self.X = np.array(
            [StochasticProcess(initial_condition=self.X_init[i])
             for i in range(self.n_nodes)])
        # Init the array of neighbors infection state processes
        self.Z = np.array(
            [StochasticProcess(initial_condition=self.Z_init[i])
             for i in range(self.n_nodes)])
        # Init the array of treatment state processes
        self.H = np.array([StochasticProcess(initial_condition=0.0)
                           for _ in range(self.n_nodes)])
        # Init the array of neighbors treatment state processes
        self.M = np.array([StochasticProcess(initial_condition=0.0)
                           for _ in range(self.n_nodes)])
        # Init the array of counting process of infections
        self.Y = np.array([CountingProcess() for _ in range(self.n_nodes)])
        # Init the array of counting process of recoveries
        self.W = np.array([CountingProcess() for _ in range(self.n_nodes)])
        # Init the array of counting process of treatments
        self.N = np.array([CountingProcess() for _ in range(self.n_nodes)])
        # Init the array of process of treatment control rates
        self.u = np.array([StochasticProcess(initial_condition=0.0)
                           for _ in range(self.n_nodes)])

        # Create infection events for initial infections
        for i in range(len(self.X_init)):
            if self.X_init[i] == 1.0:
                self.Y[i].generate_arrival_at(0.0)

        # Simulate path over time
        t = 0.0
        self.all_processes = CountingProcess()
        self.last_arrival_type = None
        
        if self.verbose:
            progress_bar = tqdm(total=self.ttotal, leave=False)

        # Plotting functionality
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
                    plt.text(0.0, self.n_nodes, s, size=12,
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
                    ax.set_ylim([0, self.n_nodes])
                    ax.set_ylabel('Number of nodes')
                    ax.legend(['Total infected |X|', 'Total under treatment |H|'])
                    plt.draw()
                    plt.pause(1e-10)

        while t < self.ttotal:

            # Compute sum of intensities
            lambdaY, lambdaW, lambdaN = self._getPoissonIntensities(t, policy_fun)
            lambda_all = np.sum(lambdaY) + np.sum(lambdaW) + np.sum(lambdaN)
            if lambda_all == 0:
                # infection went extinct
                # print("Infection went extinct at t = {}".format(round(t, 3)))
                assert(not np.any([self.X[i].value_at(t)
                                   for i in range(self.n_nodes)]))
                
                if self.verbose:
                    progress_bar.update(np.round(self.ttotal - t, 2))
               
                t = self.ttotal
                break

            # Generate next arrival of all processes Y[i], W[i], N[i]
            u = np.random.uniform(0.0, 1.0)
            w = - np.log(u) / lambda_all
            t = t + w
            self.all_processes.generate_arrival_at(t)
            
            if self.verbose:
                progress_bar.update(float(np.round(w, 2)))

            # Sample what process the arrival came from
            p = np.hstack((lambdaY, lambdaW, lambdaN)) / lambda_all
            p[p == 0.] = 0.  # sets -0.0 to 0.0
            k = np.random.choice(3 * self.n_nodes, p=p)

            # Adjust state variables accordingly
            if k < self.n_nodes:  # arrival Y
                self.last_arrival_type = 'Y'
                i = k
                self.Y[i].generate_arrival_at(t)

                # dX = dY - dW
                self.X[i].generate_arrival_at(t, 1.0)

                # dZ = A(dY - dW)
                for j in np.where(self.A[i])[0]:
                    self.Z[j].generate_arrival_at(t, 1.0)

            elif self.n_nodes <= k < 2 * self.n_nodes:  # arrival W
                self.last_arrival_type = 'W'
                i = k - self.n_nodes
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

            else:  # arrival N
                self.last_arrival_type = 'N'
                i = k - 2 * self.n_nodes
                self.N[i].generate_arrival_at(t)

                # dH = dN - H.dW
                self.H[i].generate_arrival_at(t, 1.0)

                # dM = A(dN - H.dW)
                for j in np.where(self.A[i])[0]:
                    self.M[j].generate_arrival_at(t, 1.0)

            if plot:
                callback(t)

        if plot:
            callback(t, final=True)
            plt.ioff()
            plt.show()

        if self.verbose:
            progress_bar.close()

        # return collected data for analysis
        return self._getCollectedData()

    def _getPoissonIntensities(self, t, policy_fun):
        """
        Compute and return intensities of counting processes Y, W, N as defined
        by the model.

        Parameters
        ----------
        t : float
            Time
        policy_fun : callable
            Policy function

        Returns
        -------
        lambdaY
            Intensities of the infection process Y
        lambdaW
            Intensities of the recovery process W
        lambdaN
            Intensities of the treatment process N
        """
        # Update control policy for every node
        for i, control in enumerate(policy_fun(t)):
            self.u[i].generate_arrival_at(t, None, N=control)

        # Compute intensities according to model
        lambdaY = np.array([(1 - self.X[i].value_at(t)) * (self.beta * self.Z[i].value_at(t) - self.gamma * self.M[i].value_at(t)) for i in range(self.n_nodes)])
        lambdaW = np.array([self.X[i].value_at(t) * (self.delta + self.rho * self.H[i].value_at(t)) for i in range(self.n_nodes)])
        lambdaN = np.array([self.u[i].value_at(t) * self.X[i].value_at(t) * (1 - self.H[i].value_at(t)) for i in range(self.n_nodes)])
        return lambdaY, lambdaW, lambdaN

    def simulate_policy(self, policy, baselines_dict, sim_dict, plot=False):
        """
        Simulate any given policy.

        Parameters
        ----------
        policy : str
            Name of the policy. Must be in
            - SOC    : Stochastic optimal control
            - TR     : Trivial
            - TR-FL  : Trivial (front-loaded)
            - MN     : Most-neighbors
            - MN-FL  : Most-neighbors (front-loaded)
            - LN     : Least-neighbors
            - LN-FL  : Least-neighbors (front-loaded)
            - LRSR   : Largest reduction in spectral radius policy
            - MCM    : MaxCut Minimization strategy
        baselines_dict : dict
            Scaling parameters for baselines
        sim_dict : TYPE
            Epidemic parameters for simulation
        plot : bool, optional (default: False)
            Indicate whether or not to plot stuff

        Returns
        -------
        data : dict
            Collected data from the simulation
        """
        total_time = sim_dict['total_time']
        if policy == 'SOC':
            return self._simulate(self._getOptPolicy, total_time, plot)
        elif policy == 'TR':
            return self._simulate(lambda t: self._getTrivialPolicy(baselines_dict['TR'], t), total_time, plot)
        elif policy == 'TR-FL':
            return self._simulate(lambda t: self._getTrivialPolicyFrontLoaded(baselines_dict['TR'], baselines_dict['FL_info'], t), total_time, plot)
        elif policy == 'MN':
            return self._simulate(lambda t: self._getMNDegreeHeuristicPolicy(baselines_dict['MN'], t), total_time, plot)
        elif policy == 'MN-FL':
            return self._simulate(lambda t: self._getMNDegreeHeuristicFrontLoaded(baselines_dict['MN'], baselines_dict['FL_info'], t), total_time, plot)
        elif policy == 'LN':
            return self._simulate(lambda t: self._getLNDegreeHeuristicPolicy(baselines_dict['LN'], t), total_time, plot)
        elif policy == 'LN-FL':
            return self._simulate(lambda t: self._getLNDegreeHeuristicFrontLoaded(baselines_dict['LN'], baselines_dict['FL_info'], t), total_time, plot)
        elif policy == 'LRSR':
            return self._simulate(lambda t: self._getLRSRHeuristicPolicy(baselines_dict['LRSR'], t), total_time, plot)
        elif policy == 'MCM':
            return self._simulate(lambda t: self._getMCMPolicy(baselines_dict['MCM'], t), total_time, plot)
        elif policy == 'NO':
            return self._simulate(lambda t: self._getNoPolicy(), total_time, plot)
        else:
            raise ValueError('Invalid policy name.')

    def _getCollectedData(self):
        """
        Create a dict containing all the information of the simulation for
        analysis.
        """
        return {
            'info': {
                'N': self.n_nodes,
                'beta': self.beta,
                'delta': self.delta,
                'rho': self.rho,
                'gamma': self.gamma,
                'eta': self.eta,
                'ttotal': self.ttotal,
                'Qlam': self.Qlam,
                'Qx': self.Qx,
            },
            'X': self.X,
            'H': self.H,
            'Y': self.Y,
            'W': self.W,
            'Nc': self.N,
            'u': self.u
        }

    def _getOptPolicy(self, t):
        """
        Return the stochastic optimal control policy u at time t.
        """
        # Array of X[i]'s and H[i]'s at time t
        X = np.array([self.X[i].value_at(t) for i in range(self.n_nodes)])
        H = np.array([self.H[i].value_at(t) for i in range(self.n_nodes)])

        # Only recompute if X changed at last arrival
        if (self.last_arrival_type is None or
                self.last_arrival_type == 'Y' or
                self.last_arrival_type == 'W'):

            # Convert X into array of indices where X = 0 (indices_X_is_0)
            # and X = 1 (indices_X_is_1)
            indices_X_is_0 = np.where(np.ones(self.n_nodes) - X)[0]
            indices_X_is_1 = np.where(X)[0]
            cnt_X_is_0 = indices_X_is_0.shape[0]
            cnt_X_is_1 = indices_X_is_1.shape[0]

            Qx_1 = np.delete(self.Qx, indices_X_is_0)
            Qlam_1 = np.delete(self.Qlam, indices_X_is_0)
            A_10 = np.delete(np.delete(self.A, indices_X_is_1, axis=1),
                             indices_X_is_0, axis=0)

            # Delta_V^N = (1 / K1) * [K2 - sqrt(2*K1*Qlam *
            #   (K3 * A_10.d_0 + K4) + K2^2)]
            # (see writeup)
            K1 = self.beta * (2 * self.delta + self.eta + self.rho)
            K2 = self.beta * (self.delta + self.eta) * \
                (self.delta + self.eta + self.rho) * Qlam_1
            K3 = self.eta * (self.gamma * (self. delta + self.eta) +
                            self.beta * (self.delta + self.rho))
            K4 = self.beta * (self.delta + self.rho) * Qx_1

            # LP: find d_0 s.t. slack in inequality is minimized. See appendix
            # of writeup on how to formulate LP.
            obj = np.hstack((np.ones(cnt_X_is_1), np.zeros(cnt_X_is_0)))
            epsilon = 0  # 1e-11 # to make sqrt definitely pos.; scipy.optimize.linprog tolerance is 1e-12
            epsilon_expr = epsilon / (2 * K1 * Qlam_1 * K3)
            A_ineq = np.hstack((np.zeros((cnt_X_is_1, cnt_X_is_1)), - A_10))
            b_ineq = K4 / K3 - epsilon_expr
            A_eq = np.hstack((np.eye(cnt_X_is_1), - A_10))
            b_eq = K4 / K3 - epsilon_expr

            result = scipy.optimize.linprog(obj, A_ub=A_ineq, b_ub=b_ineq,
                                            A_eq=A_eq, b_eq=b_eq)
            if result['success']:
                d_0 = result['x'][cnt_X_is_1:]
            else:
                raise Exception("LP couldn't be solved.")

            Delta_N = (1 / K1) * (K2 - np.sqrt(2 * K1 * Qlam_1 *
                                               (K3 * np.dot(A_10, d_0) + K4) + np.square(K2)))
            Delta_N_full = np.zeros(self.n_nodes)
            Delta_N_full[indices_X_is_1] = Delta_N

            # Check invariant ensured by LP: Delta_V^N is non-positive.
            if np.any(K3 * np.dot(A_10, d_0) + K4 < 0):
                raise Exception(("AssertionError: LP didn't manage to make "
                                 "sqrt() definitely greater than K2."
                                 "Consider setting epsilon > 0 in code."))

            self.last_opt = np.multiply(
                np.multiply(- Delta_N_full, 1.0 / self.Qlam),
                np.multiply(X, np.ones(self.n_nodes) - H)
            )

        return self.last_opt

    def _getTrivialPolicy(self, const, t):
        """
        Return trivial policy u at time t
        intensity(v) ~ Qx * Qlam^-1 * X * (1 - H).
        """
        # Array of X[i]'s and H[i]'s at time t
        X = np.array([self.X[i].value_at(t) for i in range(self.n_nodes)])
        H = np.array([self.H[i].value_at(t) for i in range(self.n_nodes)])
        state = np.multiply(X, np.ones(self.n_nodes) - H)
        return np.multiply(
            const * np.ones(self.n_nodes),
            np.multiply(state, self.Qx / self.Qlam))

    def _getMNDegreeHeuristicPolicy(self, const, t):
        """
        Return MN (most neighbors) degree heuristic policy u at time t
        intensity(v) ~ deg(v) * Qx * Qlam^-1 * X * (1 - H).
        """
        # Array of X[i]'s and H[i]'s at time t
        X = np.array([self.X[i].value_at(t) for i in range(self.n_nodes)])
        H = np.array([self.H[i].value_at(t) for i in range(self.n_nodes)])
        deg = np.dot(self.A.T, np.ones(self.n_nodes))
        state = np.multiply(X, np.ones(self.n_nodes) - H)
        return np.multiply(
            const * deg, np.multiply(state, self.Qx / self.Qlam))

    def _getLNDegreeHeuristicPolicy(self, const, t):
        """
        Return LN (least neighbors) degree heuristic policy u at time t
        intensity(v) ~ (maxdeg - deg(v) + 1) * Qx * Qlam^-1 * X * (1 - H).
        """
        # Array of X[i]'s and H[i]'s at time t
        X = np.array([self.X[i].value_at(t) for i in range(self.n_nodes)])
        H = np.array([self.H[i].value_at(t) for i in range(self.n_nodes)])
        deg = np.dot(self.A.T, np.ones(self.n_nodes))
        state = np.multiply(X, np.ones(self.n_nodes) - H)
        return np.multiply(
            const * ((np.max(deg) + 1) * np.ones(self.n_nodes) - deg),
            np.multiply(state, self.Qx / self.Qlam))

    def _getLRSRHeuristicPolicy(self, const, t):
        """
        Returns Largest reduction in spectral radius (LRSR) heuristic policy u
        at time t. u = 1/rank where rank is priority order of LRSR (independent
        of X(t) for consistency) const adjusts total intensity to match
        interventions with SOC.
        """
        def spectral_radius(A):
            return np.max(np.abs(np.linalg.eigvals(A)))
        if t <= 0:
            # Brute force:
            # find which node removals reduce spectral radius the most
            tau = spectral_radius(self.A)

            node_value_arr = np.zeros(self.n_nodes)
            for n in range(self.n_nodes):
                A_ = np.copy(self.A)
                A_[n, :] = np.zeros(self.n_nodes)
                A_[:, n] = np.zeros(self.n_nodes)
                node_value_arr[n] = tau - spectral_radius(A_)

            # compute ranking
            self.spectral_ranking = np.argsort(node_value_arr)

            # assign intensities according to order
            ramp = const * np.flip(np.fromfunction(
                lambda i: 1 / (i + 1), (self.n_nodes, ), dtype=float))
            u = np.ones(self.n_nodes)
            u[self.spectral_ranking] = ramp
        else:
            if self.spectral_ranking is None:
                raise Exception("Spectral radius not computed. Something went wrong.")
            ramp = const * np.flip(np.fromfunction(
                lambda i: 1/(i + 1), (self.n_nodes, ), dtype=int))
            u = np.ones(self.n_nodes)
            u[self.spectral_ranking] = ramp
        return u

    def _getMCMPolicy(self, const, t):
        """
        Return the adapted heuristic policy MaxCutMinimzation (MCM) `u` at
        time `t`. The method is adapted to fit the setup where treatment
        intensity `rho` is the equal for everyone, and the control is made on
        the rate of intervention, not the intensity of the treatment itself.
        """
        if t <= 0:
            # Find MCM priority order once and cache it
            self.mcm_ranking = maxcut.mcm(self.A)
        elif self.mcm_ranking is None:
            # The priority order should be cached. Raise an exception.
            raise Exception("MCM ranking not computed..."
                            "Something went wrong.")
        ramp = const / np.arange(self.n_nodes, 0, -1)
        u = np.ones(self.n_nodes)
        u[self.mcm_ranking] = ramp
        return u

    def _frontloadPolicy(self, u, frontload_info, t):
        """
        Return front-loaded variation of policy u at time t
        """

        # make sure limit of interventions is not reached yet
        max_count = frontload_info['N']
        interv_ct = np.sum([self.N[i].value_at(t) for i in range(self.n_nodes)])
        if interv_ct > max_count:
            u = np.zeros(self.n_nodes)

        # scale proportionally s.t. max(u) = max(u_SOC)
        if np.sum(u) > 0:

            # maximum treatment intensity of SOC over all trials
            max_SOC = frontload_info['max_u']

            # maximum treatment intensity of u over all trials
            max_u = np.max(u)

            return max_SOC * u / max_u
        else:
            return u

    def _getTrivialPolicyFrontLoaded(self, const, frontload_info, t):
        """
        Return trivial policy u at time t front-loaded to spend interventions
        earlier.
        intensity(v) ~ Qx * Qlam^-1 * X * (1 - H), at maximum intensity of SOC
        """

        # compute front-loaded variant
        u = self._getTrivialPolicy(const, t)
        return self._frontloadPolicy(u, frontload_info, t)

    def _getMNDegreeHeuristicFrontLoaded(self, const, frontload_info, t):
        """
        Returns MN (most neighbors) degree heuristic policy u at time t,
        Front-loaded: maximum total intensity of SOC over time
        is used distributed across all nodes at all times t
        """
        # compute front-loaded variant
        u = self._getMNDegreeHeuristicPolicy(const, t)
        return self._frontloadPolicy(u, frontload_info, t)

    def _getLNDegreeHeuristicFrontLoaded(self, const, frontload_info, t):
        """
        Returns LN (least neighbors) degree heuristic policy u at time t
        Front-loaded.
        """
        # compute front-loaded variant
        u = self._getLNDegreeHeuristicPolicy(const, t)
        return self._frontloadPolicy(u, frontload_info, t)

    def _getNoPolicy(self):
        """
        Return the no-policy (i.e. absence of treatment) at time t
        """
        return np.zeros(self.n_nodes)


class SIRDynamicalSystem:
    """
    Class that implements the simulation of the disease control dynamical
    system.
    """

    def __init__(self, X_init, A, param, cost, verbose=True, debug=False):
        self.n_nodes = A.shape[0]  # Number of nodes
        self.beta = param['beta']  # Infection rate
        self.gamma = param['gamma']  # Reduc.in infection rate from treatment
        self.delta = param['delta']  # Recovery rate (spontaneous)
        self.rho = param['rho']  # Recovery rate from treatment
        self.eta = param['eta']  # Exponential discount rate for SOC strategy

        # LRSR
        self.spectral_ranking = None

        # MCM
        self.mcm_ranking = None

        self.G = nx.from_numpy_matrix(A, parallel_edges=False, create_using=nx.Graph)

        self.verbose = verbose
        self.debug = debug

        if (len(X_init) == self.n_nodes and (A.shape[0] == A.shape[1])):
            self.A = A
            self.X_init = X_init
            self.Z_init = np.dot(A.T, X_init)
            self.Qlam = cost['Qlam'] * np.ones(self.n_nodes)
            self.Qx = cost['Qx'] * np.ones(self.n_nodes)
        else:
            raise ValueError("Dimensions don't match")

    def _simulate(self, policy_fun, time, plot, plot_update=1.0):
        """
        Simulate the SIS dynamical system using Ogata's thinning algorithm over
        the time period policy_fun must be of shape: (1,) -> (N,)
        where t in [0, T] -> control vector for all N nodes.
        """

        # Initialize the end time of the simulation
        self.ttotal = time

        # Init the array of infection state processes
        self.X = np.array(
            [StochasticProcess(initial_condition=self.X_init[i])
             for i in range(self.n_nodes)])
        # Init the array of neighbors infection state processes
        self.Z = np.array(
            [StochasticProcess(initial_condition=self.Z_init[i])
             for i in range(self.n_nodes)])
        # Init the array of treatment state processes
        self.H = np.array([StochasticProcess(initial_condition=0.0)
                           for _ in range(self.n_nodes)])
        # Init the array of neighbors treatment state processes
        self.M = np.array([StochasticProcess(initial_condition=0.0)
                           for _ in range(self.n_nodes)])
        # Init the array of counting process of infections
        self.Y = np.array([CountingProcess() for _ in range(self.n_nodes)])
        # Init the array of counting process of recoveries
        self.W = np.array([CountingProcess() for _ in range(self.n_nodes)])
        # Init the array of counting process of treatments
        self.N = np.array([CountingProcess() for _ in range(self.n_nodes)])
        # Init the array of process of treatment control rates
        self.u = np.array([StochasticProcess(initial_condition=0.0)
                           for _ in range(self.n_nodes)])

        # Create infection events for initial infections
        for i in range(len(self.X_init)):
            if self.X_init[i] == 1.0:
                self.Y[i].generate_arrival_at(0.0)

        # Simulate path over time
        t = 0.0
        self.all_processes = CountingProcess()
        self.last_arrival_type = None
        
        if self.verbose:
            progress_bar = tqdm(total=self.ttotal, leave=False)

        # Plotting functionality
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
                    plt.text(0.0, self.n_nodes, s, size=12,
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
                    ax.set_ylim([0, self.n_nodes])
                    ax.set_ylabel('Number of nodes')
                    ax.legend(['Total infected |X|', 'Total under treatment |H|'])
                    plt.draw()
                    plt.pause(1e-10)

        while t < self.ttotal:

            # Compute sum of intensities
            lambdaY, lambdaW, lambdaN = self._getPoissonIntensities(t, policy_fun)
            lambda_all = np.sum(lambdaY) + np.sum(lambdaW) + np.sum(lambdaN)

            if self.debug:
                print("\n------")
                print(f"t={t:.4f}")
                print((f"lamY={lambdaY.sum():.2f}, "
                       f"lamW={lambdaW.sum():.2f}, "
                       f"lamN={lambdaN.sum():.2f}, "
                       f"lamSUM={lambda_all:.2f}"))
                Y = np.array([self.Y[i].value_at(t) for i in range(self.n_nodes)])
                W = np.array([self.W[i].value_at(t) for i in range(self.n_nodes)])
                H = np.array([self.H[i].value_at(t) for i in range(self.n_nodes)])
                assert Y.max() <= 1, "Y should have values at most 1!"
                assert W.max() <= 1, "Y should have values at most 1!"
                assert H.max() <= 1, "Y should have values at most 1!"
                Y = Y.astype(bool)
                W = W.astype(bool)
                H = H.astype(bool)
                print('Susceptible nodes:', np.where(~Y & ~W)[0])
                print('Infected nodes:', np.where(Y & ~W)[0])
                print('Recovered nodes:', np.where(Y & W)[0])
                print('Treated nodes:', np.where(H)[0])
                assert np.sum(~Y & W) == 0, "There are recovered but not infected nodes!"
            
            # Infection went extinct
            if lambda_all == 0:
                # print("Infection went extinct at t = {}".format(round(t, 3)))
                assert(not np.any([self.X[i].value_at(t)
                                   for i in range(self.n_nodes)]))
                
                if self.verbose:
                    progress_bar.update(np.round(self.ttotal - t, 2))
               
                t = self.ttotal
                break

            # Generate next arrival of all processes Y[i], W[i], N[i]
            u = np.random.uniform(0.0, 1.0)
            w = - np.log(u) / lambda_all
            t = t + w
            self.all_processes.generate_arrival_at(t)

            print(f"t+w={t:.4f}")
            
            if self.verbose:
                progress_bar.update(float(np.round(w, 2)))

            # Sample what process the arrival came from
            p = np.hstack((lambdaY, lambdaW, lambdaN)) / lambda_all
            p[p == 0.] = 0.  # sets -0.0 to 0.0
            k = np.random.choice(3 * self.n_nodes, p=p)

            # Adjust state variables accordingly
            if k < self.n_nodes:  # arrival Y (infection)
                self.last_arrival_type = 'Y'
                i = k

                print(f'Process INFECTION event at node {i:d}')

                self.Y[i].generate_arrival_at(t)

                # dX = dY - dW
                self.X[i].generate_arrival_at(t, 1.0)

                # dZ = A(dY - dW)
                for j in np.where(self.A[i])[0]:
                    self.Z[j].generate_arrival_at(t, 1.0)

            elif self.n_nodes <= k < 2 * self.n_nodes:  # arrival W (recovery)
                self.last_arrival_type = 'W'
                i = k - self.n_nodes

                print(f'Process RECOVERY event at node {i:d}')

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

            else:  # arrival N (treatment)
                self.last_arrival_type = 'N'
                i = k - 2 * self.n_nodes

                print(f'Process TREATMENT event at node {i:d}')

                self.N[i].generate_arrival_at(t)

                # dH = dN - H.dW
                self.H[i].generate_arrival_at(t, 1.0)

                # dM = A(dN - H.dW)
                for j in np.where(self.A[i])[0]:
                    self.M[j].generate_arrival_at(t, 1.0)

            if plot:
                callback(t)

        if plot:
            callback(t, final=True)
            plt.ioff()
            plt.show()

        if self.verbose:
            progress_bar.close()

        # return collected data for analysis
        return self._getCollectedData()

    def _getPoissonIntensities(self, t, policy_fun):
        """
        Compute and return intensities of counting processes Y, W, N as defined
        by the model.

        Parameters
        ----------
        t : float
            Time
        policy_fun : callable
            Policy function

        Returns
        -------
        lambdaY
            Intensities of the infection process Y
        lambdaW
            Intensities of the recovery process W
        lambdaN
            Intensities of the treatment process N
        """
        # Update control policy for every node
        for i, control in enumerate(policy_fun(t)):
            self.u[i].generate_arrival_at(t, None, N=control)

        # Compute intensities according to model
        lambdaY = np.array([(1 - self.Y[i].value_at(t)) * (self.beta * self.Z[i].value_at(t) - self.gamma * self.M[i].value_at(t)) for i in range(self.n_nodes)])
        lambdaW = np.array([self.X[i].value_at(t) * (self.delta + self.rho * self.H[i].value_at(t)) for i in range(self.n_nodes)])
        lambdaN = np.array([self.u[i].value_at(t) * self.X[i].value_at(t) * (1 - self.H[i].value_at(t)) for i in range(self.n_nodes)])
        return lambdaY, lambdaW, lambdaN

    def simulate_policy(self, policy, baselines_dict, sim_dict, plot=False):
        """
        Simulate any given policy.

        Parameters
        ----------
        policy : str
            Name of the policy. Must be in
            - SOC    : Stochastic optimal control
            - TR     : Trivial
            - TR-FL  : Trivial (front-loaded)
            - MN     : Most-neighbors
            - MN-FL  : Most-neighbors (front-loaded)
            - LN     : Least-neighbors
            - LN-FL  : Least-neighbors (front-loaded)
            - LRSR   : Largest reduction in spectral radius policy
            - MCM    : MaxCut Minimization strategy
        baselines_dict : dict
            Scaling parameters for baselines
        sim_dict : TYPE
            Epidemic parameters for simulation
        plot : bool, optional (default: False)
            Indicate whether or not to plot stuff

        Returns
        -------
        data : dict
            Collected data from the simulation
        """
        total_time = sim_dict['total_time']
        if policy == 'SOC':
            return self._simulate(self._getOptPolicy, total_time, plot)
        elif policy == 'TR':
            return self._simulate(lambda t: self._getTrivialPolicy(baselines_dict['TR'], t), total_time, plot)
        elif policy == 'TR-FL':
            return self._simulate(lambda t: self._getTrivialPolicyFrontLoaded(baselines_dict['TR'], baselines_dict['FL_info'], t), total_time, plot)
        elif policy == 'MN':
            return self._simulate(lambda t: self._getMNDegreeHeuristicPolicy(baselines_dict['MN'], t), total_time, plot)
        elif policy == 'MN-FL':
            return self._simulate(lambda t: self._getMNDegreeHeuristicFrontLoaded(baselines_dict['MN'], baselines_dict['FL_info'], t), total_time, plot)
        elif policy == 'LN':
            return self._simulate(lambda t: self._getLNDegreeHeuristicPolicy(baselines_dict['LN'], t), total_time, plot)
        elif policy == 'LN-FL':
            return self._simulate(lambda t: self._getLNDegreeHeuristicFrontLoaded(baselines_dict['LN'], baselines_dict['FL_info'], t), total_time, plot)
        elif policy == 'LRSR':
            return self._simulate(lambda t: self._getLRSRHeuristicPolicy(baselines_dict['LRSR'], t), total_time, plot)
        elif policy == 'MCM':
            return self._simulate(lambda t: self._getMCMPolicy(baselines_dict['MCM'], t), total_time, plot)
        elif policy == 'NO':
            return self._simulate(lambda t: self._getNoPolicy(), total_time, plot)
        else:
            raise ValueError('Invalid policy name.')

    def _getCollectedData(self):
        """
        Create a dict containing all the information of the simulation for
        analysis.
        """
        return {
            'info': {
                'N': self.n_nodes,
                'beta': self.beta,
                'delta': self.delta,
                'rho': self.rho,
                'gamma': self.gamma,
                'eta': self.eta,
                'ttotal': self.ttotal,
                'Qlam': self.Qlam,
                'Qx': self.Qx,
            },
            'X': self.X,
            'H': self.H,
            'Y': self.Y,
            'W': self.W,
            'Nc': self.N,
            'u': self.u
        }

    def _getOptPolicy(self, t):
        """
        Return the stochastic optimal control policy u at time t.
        """
        # Only recompute if X changed at last arrival
        if self.last_arrival_type in [None, 'Y', 'W']:
            # Array of X[i]'s, Y[i]'s and H[i]'s at time t
            X = np.array([self.X[i].value_at(t) for i in range(self.n_nodes)])
            Y = np.array([self.Y[i].value_at(t) for i in range(self.n_nodes)])
            H = np.array([self.H[i].value_at(t) for i in range(self.n_nodes)])

            # Convert X into array of indices where X = 0 (indices_X_is_0)
            # and X = 1 (indices_X_is_1)
            indices_X_is_0 = np.where(np.ones(self.n_nodes) - X)[0]
            indices_X_is_1 = np.where(X)[0]
            cnt_X_is_0 = indices_X_is_0.shape[0]
            cnt_X_is_1 = indices_X_is_1.shape[0]

            Qx_1 = np.delete(self.Qx, indices_X_is_0)
            Qlam_1 = np.delete(self.Qlam, indices_X_is_0)
            A_10 = self.A[indices_X_is_1[:, np.newaxis], indices_X_is_0]

            # Delta_V^N = (1 / K1) * [K2 - sqrt(2*K1*Qlam *
            #   (K3 * A_10.d_0 + K4) + K2^2)]
            # (see writeup)
            K1 = self.beta * (2 * self.delta + self.eta + self.rho)
            K2 = self.beta * (self.delta + self.eta) * \
                (self.delta + self.eta + self.rho) * Qlam_1
            K3 = self.eta * (self.gamma * (self. delta + self.eta) +
                             self.beta * (self.delta + self.rho))
            K4 = self.beta * (self.delta + self.rho) * Qx_1

            # LP: find d_0 s.t. slack in inequality is minimized. See appendix
            # of writeup on how to formulate LP.
            obj = np.hstack((np.ones(cnt_X_is_1), np.zeros(cnt_X_is_0)))
            # small epsilon to make sqrt definitely pos.
            # scipy.optimize.linprog tolerance is 1e-12
            epsilon = 0
            epsilon_expr = epsilon / (2 * K1 * Qlam_1 * K3)
            A_ineq = np.hstack((np.zeros((cnt_X_is_1, cnt_X_is_1)), - A_10))
            b_ineq = K4 / K3 - epsilon_expr
            A_eq = np.hstack((np.eye(cnt_X_is_1), - A_10))
            b_eq = K4 / K3 - epsilon_expr

            result = scipy.optimize.linprog(obj, A_ub=A_ineq, b_ub=b_ineq,
                                            A_eq=A_eq, b_eq=b_eq)
            if result['success']:
                d_0 = result['x'][cnt_X_is_1:]
            else:
                raise Exception("LP couldn't be solved.")

            Delta_N = (1 / K1) * (K2 - np.sqrt(
                2 * K1 * Qlam_1 * (K3 * np.dot(A_10, d_0) + K4) + K2 ** 2
            ))
            Delta_N_full = np.zeros(self.n_nodes)
            Delta_N_full[indices_X_is_1] = Delta_N

            # Check invariant ensured by LP: Delta_V^N is non-positive.
            if np.any(K3 * np.dot(A_10, d_0) + K4 < 0):
                raise Exception(("AssertionError: LP didn't manage to make "
                                 "sqrt() definitely greater than K2."
                                 "Consider setting epsilon > 0 in code."))

            self.last_opt = - X * Y * Delta_N_full * (1 - H) / self.Qlam

        return self.last_opt

    def _getTrivialPolicy(self, const, t):
        """
        Return trivial policy u at time t
        intensity(v) ~ Qx * Qlam^-1 * X * (1 - H).
        """
        # Array of X[i]'s and H[i]'s at time t
        X = np.array([self.X[i].value_at(t) for i in range(self.n_nodes)])
        H = np.array([self.H[i].value_at(t) for i in range(self.n_nodes)])
        state = np.multiply(X, np.ones(self.n_nodes) - H)
        return np.multiply(
            const * np.ones(self.n_nodes),
            np.multiply(state, self.Qx / self.Qlam))

    def _getMNDegreeHeuristicPolicy(self, const, t):
        """
        Return MN (most neighbors) degree heuristic policy u at time t
        intensity(v) ~ deg(v) * Qx * Qlam^-1 * X * (1 - H).
        """
        # Array of X[i]'s and H[i]'s at time t
        X = np.array([self.X[i].value_at(t) for i in range(self.n_nodes)])
        H = np.array([self.H[i].value_at(t) for i in range(self.n_nodes)])
        deg = np.dot(self.A.T, np.ones(self.n_nodes))
        state = np.multiply(X, np.ones(self.n_nodes) - H)
        return np.multiply(
            const * deg, np.multiply(state, self.Qx / self.Qlam))

    def _getLNDegreeHeuristicPolicy(self, const, t):
        """
        Return LN (least neighbors) degree heuristic policy u at time t
        intensity(v) ~ (maxdeg - deg(v) + 1) * Qx * Qlam^-1 * X * (1 - H).
        """
        # Array of X[i]'s and H[i]'s at time t
        X = np.array([self.X[i].value_at(t) for i in range(self.n_nodes)])
        H = np.array([self.H[i].value_at(t) for i in range(self.n_nodes)])
        deg = np.dot(self.A.T, np.ones(self.n_nodes))
        state = np.multiply(X, np.ones(self.n_nodes) - H)
        return np.multiply(
            const * ((np.max(deg) + 1) * np.ones(self.n_nodes) - deg),
            np.multiply(state, self.Qx / self.Qlam))

    def _getLRSRHeuristicPolicy(self, const, t):
        """
        Returns Largest reduction in spectral radius (LRSR) heuristic policy u
        at time t. u = 1/rank where rank is priority order of LRSR (independent
        of X(t) for consistency) const adjusts total intensity to match
        interventions with SOC.
        """
        def spectral_radius(A):
            return np.max(np.abs(np.linalg.eigvals(A)))
        if t <= 0:
            # Brute force:
            # find which node removals reduce spectral radius the most
            tau = spectral_radius(self.A)

            node_value_arr = np.zeros(self.n_nodes)
            for n in range(self.n_nodes):
                A_ = np.copy(self.A)
                A_[n, :] = np.zeros(self.n_nodes)
                A_[:, n] = np.zeros(self.n_nodes)
                node_value_arr[n] = tau - spectral_radius(A_)

            # compute ranking
            self.spectral_ranking = np.argsort(node_value_arr)

            # assign intensities according to order
            ramp = const * np.flip(np.fromfunction(
                lambda i: 1 / (i + 1), (self.n_nodes, ), dtype=float))
            u = np.ones(self.n_nodes)
            u[self.spectral_ranking] = ramp
        else:
            if self.spectral_ranking is None:
                raise Exception("Spectral radius not computed. Something went wrong.")
            ramp = const * np.flip(np.fromfunction(
                lambda i: 1/(i + 1), (self.n_nodes, ), dtype=int))
            u = np.ones(self.n_nodes)
            u[self.spectral_ranking] = ramp
        return u

    def _getMCMPolicy(self, const, t):
        """
        Return the adapted heuristic policy MaxCutMinimzation (MCM) `u` at
        time `t`. The method is adapted to fit the setup where treatment
        intensity `rho` is the equal for everyone, and the control is made on
        the rate of intervention, not the intensity of the treatment itself.
        """
        if t <= 0:
            # Find MCM priority order once and cache it
            self.mcm_ranking = maxcut.mcm(self.A)
        elif self.mcm_ranking is None:
            # The priority order should be cached. Raise an exception.
            raise Exception("MCM ranking not computed..."
                            "Something went wrong.")
        ramp = const / np.arange(self.n_nodes, 0, -1)
        u = np.ones(self.n_nodes)
        u[self.mcm_ranking] = ramp
        return u

    def _frontloadPolicy(self, u, frontload_info, t):
        """
        Return front-loaded variation of policy u at time t
        """

        # make sure limit of interventions is not reached yet
        max_count = frontload_info['N']
        interv_ct = np.sum([self.N[i].value_at(t) for i in range(self.n_nodes)])
        if interv_ct > max_count:
            u = np.zeros(self.n_nodes)

        # scale proportionally s.t. max(u) = max(u_SOC)
        if np.sum(u) > 0:

            # maximum treatment intensity of SOC over all trials
            max_SOC = frontload_info['max_u']

            # maximum treatment intensity of u over all trials
            max_u = np.max(u)

            return max_SOC * u / max_u
        else:
            return u

    def _getTrivialPolicyFrontLoaded(self, const, frontload_info, t):
        """
        Return trivial policy u at time t front-loaded to spend interventions
        earlier.
        intensity(v) ~ Qx * Qlam^-1 * X * (1 - H), at maximum intensity of SOC
        """

        # compute front-loaded variant
        u = self._getTrivialPolicy(const, t)
        return self._frontloadPolicy(u, frontload_info, t)

    def _getMNDegreeHeuristicFrontLoaded(self, const, frontload_info, t):
        """
        Returns MN (most neighbors) degree heuristic policy u at time t,
        Front-loaded: maximum total intensity of SOC over time
        is used distributed across all nodes at all times t
        """
        # compute front-loaded variant
        u = self._getMNDegreeHeuristicPolicy(const, t)
        return self._frontloadPolicy(u, frontload_info, t)

    def _getLNDegreeHeuristicFrontLoaded(self, const, frontload_info, t):
        """
        Returns LN (least neighbors) degree heuristic policy u at time t
        Front-loaded.
        """
        # compute front-loaded variant
        u = self._getLNDegreeHeuristicPolicy(const, t)
        return self._frontloadPolicy(u, frontload_info, t)

    def _getNoPolicy(self):
        """
        Return the no-policy (i.e. absence of treatment) at time t
        """
        return np.zeros(self.n_nodes)
