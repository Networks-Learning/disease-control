
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, tqdm_notebook
import scipy.optimize
import networkx as nx

from stochastic_processes import StochasticProcess, CountingProcess
from helpers import HelperFunc
import maxcut


class Monitor:

    def __init__(self, verbose, use_tqdm):
        self.verbose = verbose
        self.use_tqdm = use_tqdm
        self.time_elapsed = 0.0

    def start(self, system):
        if self.verbose:
            if self.use_tqdm:
                progress_bar = tqdm(total=system.max_time, leave=False)

    def print(self, system, w):
        if self.verbose:
            if self.use_tqdm:
                progress_bar.update(float(np.round(w, 2)))
            else:
                self.time_elapsed += w
                N = np.array([self.N[i].value_at(self.time_elapsed) for i in range(self.n_nodes)])
                print(f"\rtime {self.time_elapsed:>6.2f}/{system.max_time:<6.2f} | "
                      f"S: {system.num_suc:>4d}, I:{system.num_inf:>4d}, "
                      f"H: {system.num_treated:>4d}"
                      f"N: {N.sum():>4d}"
                      "           ", end='')

    def stop(self, system):
        if self.verbose:
            if self.use_tqdm:
                progress_bar.close()
            else:
                N = np.array([self.N[i].value_at(self.time_elapsed) for i in range(self.n_nodes)])
                print(f"\rtime {self.time_elapsed:>6.2f}/{system.max_time:<6.2f} | "
                      f"S: {system.num_suc:>4d}, I:{system.num_inf:>4d}, "
                      f"H: {system.num_treated:>4d}"
                      f"N: {N.sum():>4d}"
                      "           ", end='\n')


class SISDynamicalSystem:
    """
    Class that implements the simulation of the disease control dynamical
    system.
    """

    def __init__(self, graph, param, cost, min_d0=0.0, verbose=False, notebook=False, debug=False):
        if param['gamma'] > param['beta']:
            raise ValueError("`beta` must be larger than `gamma`!")
        if min(list(param.values())) < 0:
            raise ValueError("Epidemic parameters must be non-negative!")
        self.beta = param['beta']  # Infection rate
        self.gamma = param['gamma']  # Reduc.in infection rate from treatment
        self.delta = param['delta']  # Recovery rate (spontaneous)
        self.rho = param['rho']  # Recovery rate from treatment
        self.eta = param['eta']  # Exponential discount rate for SOC strategy
        
        self.spectral_ranking = None  # LRSR
        self.mcm_ranking = None  # MCM

        self.debug_d0 = {'min': np.inf, 'max': -np.inf}
        self.min_d0 = min_d0

        self.G = graph
        self.n_nodes = graph.number_of_nodes()  # Number of nodes

        self.debug = debug
        self.monitor = Monitor(verbose=verbose, use_tqdm=not notebook)
        
        self.A = nx.adjacency_matrix(self.G).toarray().astype(float)
        self.Qlam = cost['Qlam'] * np.ones(self.n_nodes)
        self.Qx = cost['Qx'] * np.ones(self.n_nodes)

    def _simulate(self, policy_fun, X_init, max_time):
        """
        Simulate the SIS dynamical system using Ogata's thinning algorithm over
        the time period policy_fun must be of shape: (1,) -> (N,)
        where t in [0, T] -> control vector for all N nodes.
        """

        if X_init.shape != (self.n_nodes,):
            raise ValueError("Invalid shape of `X_init`. Should be (`n_nodes`,)")
        Z_init = self.A.T.dot(X_init)

        # Initialize the end time of the simulation
        self.max_time = max_time

        # Init the array of infection state processes
        self.X = np.array([StochasticProcess(init_value=X_init[i]) for i in range(self.n_nodes)])
        # Init the array of neighbors infection state processes
        self.Z = np.array([StochasticProcess(init_value=Z_init[i]) for i in range(self.n_nodes)])
        # Init the array of treatment state processes
        self.H = np.array([StochasticProcess(init_value=0.0) for _ in range(self.n_nodes)])
        # Init the array of neighbors treatment state processes
        self.M = np.array([StochasticProcess(init_value=0.0) for _ in range(self.n_nodes)])
        # Init the array of counting process of infections
        self.Y = np.array([CountingProcess() for _ in range(self.n_nodes)])
        # Init the array of counting process of recoveries
        self.W = np.array([CountingProcess() for _ in range(self.n_nodes)])
        # Init the array of counting process of treatments
        self.N = np.array([CountingProcess() for _ in range(self.n_nodes)])
        # Init the array of process of treatment control rates
        self.u = np.array([StochasticProcess(init_value=0.0)  for _ in range(self.n_nodes)])

        # Create infection events for initial infections
        for i in np.where(X_init)[0]:
            self.Y[i].generate_arrival_at(0.0)

        # Simulate path over time
        t = 0.0
        self.all_processes = CountingProcess()
        self.last_arrival_type = None
        
        self.monitor.start(self)

        self.num_inf = int(X_init.sum())
        self.num_suc = self.n_nodes - self.num_inf
        self.num_treated = 0

        while t < self.max_time:

            # Compute sum of intensities
            lambdaY, lambdaW, lambdaN = self._getPoissonIntensities(t, policy_fun)
            lambda_all = np.sum(lambdaY) + np.sum(lambdaW) + np.sum(lambdaN)
            if lambda_all == 0:
                # infection went extinct
                X = [self.X[i].value_at(t) for i in range(self.n_nodes)]
                assert not np.any(X)  # Sanity check that there are no infections
                
                self.monitor.stop(self)
               
                t = self.max_time
                break

            # Generate next arrival of all processes Y[i], W[i], N[i]
            u = np.random.uniform(0.0, 1.0)
            w = - np.log(u) / lambda_all
            t = t + w
            self.all_processes.generate_arrival_at(t)

            self.monitor.print(self, w)

            # Sample what process the arrival came from
            p = np.hstack((lambdaY, lambdaW, lambdaN)) / lambda_all
            p[p == 0.] = 0.  # sets -0.0 to 0.0
            k = np.random.choice(3 * self.n_nodes, p=p)

            # Adjust state variables accordingly
            if k < self.n_nodes:  # arrival Y
                self.last_arrival_type = 'Y'
                i = k

                self.num_suc -= 1
                self.num_inf += 1

                self.Y[i].generate_arrival_at(t)

                # dX = dY - dW
                self.X[i].generate_arrival_at(t, 1.0)

                # dZ = A(dY - dW)
                for j in np.where(self.A[i])[0]:
                    self.Z[j].generate_arrival_at(t, 1.0)

            elif self.n_nodes <= k < 2 * self.n_nodes:  # arrival W
                self.last_arrival_type = 'W'
                i = k - self.n_nodes
                
                self.num_suc += 1
                self.num_inf -= 1

                self.W[i].generate_arrival_at(t)

                # dX = dY - dW
                self.X[i].generate_arrival_at(t, -1.0)

                # dZ = A(dY - dW)
                for j in np.where(self.A[i])[0]:
                    self.Z[j].generate_arrival_at(t, -1.0)

                
                prev_H = self.H[i].value_at(t)
                if prev_H == 1:
                    # dH = dN - H.dW
                    self.H[i].generate_arrival_at(t, -1.0)
                    # dM = A(dN - H.dW)
                    for j in np.where(self.A[i])[0]:
                        self.M[j].generate_arrival_at(t, -1.0)
                    self.num_treated -= 1

            else:  # arrival N
                self.last_arrival_type = 'N'
                i = k - 2 * self.n_nodes

                self.num_treated += 1

                self.N[i].generate_arrival_at(t)

                # dH = dN - H.dW
                self.H[i].generate_arrival_at(t, 1.0)

                # dM = A(dN - H.dW)
                for j in np.where(self.A[i])[0]:
                    self.M[j].generate_arrival_at(t, 1.0)

        self.monitor.stop(self)

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

    def simulate_policy(self, policy, X_init, max_time, baselines_dict):
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

        Returns
        -------
        data : dict
            Collected data from the simulation
        """
        if policy == 'SOC':
            return self._simulate(self._getOptPolicy, X_init, max_time)
        elif policy == 'TR':
            return self._simulate(lambda t: self._getTrivialPolicy(baselines_dict['TR'], t),  X_init, max_time)
        elif policy == 'TR-FL':
            return self._simulate(lambda t: self._getTrivialPolicyFrontLoaded(baselines_dict['TR'], baselines_dict['FL_info'], t), X_init, max_time)
        elif policy == 'MN':
            return self._simulate(lambda t: self._getMNDegreeHeuristicPolicy(baselines_dict['MN'], t), X_init, max_time)
        elif policy == 'MN-FL':
            return self._simulate(lambda t: self._getMNDegreeHeuristicFrontLoaded(baselines_dict['MN'], baselines_dict['FL_info'], t),  X_init, max_time)
        elif policy == 'LN':
            return self._simulate(lambda t: self._getLNDegreeHeuristicPolicy(baselines_dict['LN'], t), X_init, max_time)
        elif policy == 'LN-FL':
            return self._simulate(lambda t: self._getLNDegreeHeuristicFrontLoaded(baselines_dict['LN'], baselines_dict['FL_info'], t), X_init, max_time)
        elif policy == 'LRSR':
            return self._simulate(lambda t: self._getLRSRHeuristicPolicy(baselines_dict['LRSR'], t), X_init, max_time)
        elif policy == 'MCM':
            return self._simulate(lambda t: self._getMCMPolicy(baselines_dict['MCM'], t),  X_init, max_time)
        elif policy == 'NO':
            return self._simulate(lambda t: self._getNoPolicy(),  X_init, max_time)
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
                'max_time': self.max_time,
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
            epsilon = 1e-8  # 1e-11 # to make sqrt definitely pos.; scipy.optimize.linprog tolerance is 1e-12
            epsilon_expr = epsilon / (2 * K1 * Qlam_1 * K3)
            A_ineq = np.hstack((np.zeros((cnt_X_is_1, cnt_X_is_1)), - A_10))
            b_ineq = K4 / K3 - epsilon_expr
            A_eq = np.hstack((np.eye(cnt_X_is_1), - A_10))
            b_eq = K4 / K3 - epsilon_expr

            result = scipy.optimize.linprog(obj, A_ub=A_ineq, b_ub=b_ineq,
                                            A_eq=A_eq, b_eq=b_eq,
                                            bounds=(self.min_d0, None),
                                            options={'tol': 1e-8})
            if result['success']:
                d_0 = result['x'][cnt_X_is_1:]
                if cnt_X_is_0 > 0:
                    self.debug_d0['min'] = min(self.debug_d0['min'], d_0.min())
                    self.debug_d0['max'] = max(self.debug_d0['max'], d_0.max())
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
        state = X * (1 - H)
        return  state * const

    def _getMNDegreeHeuristicPolicy(self, const, t):
        """
        Return MN (most neighbors) degree heuristic policy u at time t
        intensity(v) ~ deg(v) * Qx * Qlam^-1 * X * (1 - H).
        """
        # Array of X[i]'s and H[i]'s at time t
        X = np.array([self.X[i].value_at(t) for i in range(self.n_nodes)])
        H = np.array([self.H[i].value_at(t) for i in range(self.n_nodes)])
        state = X * (1 - H)
        deg = np.dot(self.A.T, np.ones(self.n_nodes))
        return deg * state * const

    def _getLNDegreeHeuristicPolicy(self, const, t):
        """
        Return LN (least neighbors) degree heuristic policy u at time t
        intensity(v) ~ (maxdeg - deg(v) + 1) * Qx * Qlam^-1 * X * (1 - H).
        """
        # Array of X[i]'s and H[i]'s at time t
        X = np.array([self.X[i].value_at(t) for i in range(self.n_nodes)])
        H = np.array([self.H[i].value_at(t) for i in range(self.n_nodes)])
        deg = self.A.T.dot(np.ones(self.n_nodes))
        prop = np.max(deg) + 1 - deg
        state = X * (1 - H)
        return prop * state * const

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

    def __init__(self, X_init, graph, param, cost, min_d0=0.0, verbose=True, debug=False, notebook=False):
        if self.gamma > self.beta:
            raise ValueError("`beta` must be larger than `gamma`!")
        if min(self.beta, self.gamma, self.delta, self.rho, self.eta) < 0:
            raise ValueError("Epidemic parameters must be non-negative!")
        self.beta = param['beta']  # Infection rate
        self.gamma = param['gamma']  # Reduc.in infection rate from treatment
        self.delta = param['delta']  # Recovery rate (spontaneous)
        self.rho = param['rho']  # Recovery rate from treatment
        self.eta = param['eta']  # Exponential discount rate for SOC strategy
        
        self.spectral_ranking = None  # LRSR
        self.mcm_ranking = None  # MCM

        self.min_d0 = min_d0

        self.G = graph
        self.n_nodes = graph.number_of_nodes()  # Number of nodes

        self.verbose = verbose
        self.debug = debug
        self.notebook = notebook

        if len(X_init) != self.n_nodes:
            raise ValueError("Dimensions don't match")
        
        self.A = nx.adjacency_matrix(self.G)
        self.X_init = X_init
        self.Z_init = self.A.T.dot(X_init)
        self.Qlam = cost['Qlam'] * np.ones(self.n_nodes)
        self.Qx = cost['Qx'] * np.ones(self.n_nodes)
            

    def _simulate(self, policy_fun, total_time, plot, plot_update=1.0):
        """
        Simulate the SIS dynamical system using Ogata's thinning algorithm over
        the time period policy_fun must be of shape: (1,) -> (N,)
        where t in [0, T] -> control vector for all N nodes.
        """

        # Initialize the end time of the simulation
        self.ttotal = total_time

        # Init the array of infection state processes
        self.X = np.array(
            [StochasticProcess(init_value=self.X_init[i])
             for i in range(self.n_nodes)])
        # Init the array of neighbors infection state processes
        self.Z = np.array(
            [StochasticProcess(init_value=self.Z_init[i])
             for i in range(self.n_nodes)])
        # Init the array of treatment state processes
        self.H = np.array([StochasticProcess(init_value=0.0)
                           for _ in range(self.n_nodes)])
        # Init the array of neighbors treatment state processes
        self.M = np.array([StochasticProcess(init_value=0.0)
                           for _ in range(self.n_nodes)])
        # Init the array of counting process of infections
        self.Y = np.array([CountingProcess() for _ in range(self.n_nodes)])
        # Init the array of counting process of recoveries
        self.W = np.array([CountingProcess() for _ in range(self.n_nodes)])
        # Init the array of counting process of treatments
        self.N = np.array([CountingProcess() for _ in range(self.n_nodes)])
        # Init the array of process of treatment control rates
        self.u = np.array([StochasticProcess(init_value=0.0)
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
            if self.notebook:
                pass
            else: 
                progress_bar = tqdm(total=self.ttotal, leave=False)

        self.num_inf = int(np.array([self.Y[i].value_at(t) for i in range(self.n_nodes)]).sum())
        self.num_suc = self.n_nodes - self.num_inf
        self.num_rec = 0
        self.num_treated = 0
        last_time = time.time()

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
                print('Susceptible nodes:', len(np.where(~Y & ~W)[0]))
                print('Infected nodes:', len(np.where(Y & ~W)[0]))
                print('Recovered nodes:', len(np.where(Y & W)[0]))
                print('Treated nodes:', len(np.where(H)[0]))
                assert np.sum(~Y & W) == 0, "There are recovered but not infected nodes!"
            
            # Infection went extinct
            if lambda_all == 0:
                # print("Infection went extinct at t = {}".format(round(t, 3)))
                assert(not np.any([self.X[i].value_at(t)
                                   for i in range(self.n_nodes)]))
                
                if self.verbose:
                    if self.notebook:
                        this_time = time.time()
                        iter_speed = 1 / (this_time - last_time)
                        last_time = this_time
                        print(f"\rtime {t:>6.2f}/{self.ttotal:<6.2f} | "
                              f"S: {self.num_suc:>4d}, I:{self.num_inf:>4d}, "
                              f"R: {self.num_rec:d}, H: {self.num_treated:d}, "
                              f"lY: {lambdaY.sum():.2f}, lW: {lambdaW.sum():.2f}, lN: {lambdaN.sum():.2f} | "
                              f"{iter_speed:.2f} iter/s"
                              "           ", 
                              end='')
                    else:
                        progress_bar.update(np.round(self.ttotal - t, 2))
               
                t = self.ttotal
                break

            # Generate next arrival of all processes Y[i], W[i], N[i]
            u = np.random.uniform(0.0, 1.0)
            w = - np.log(u) / lambda_all
            t = t + w
            self.all_processes.generate_arrival_at(t)

            if self.debug:
                print(f"t+w={t:.4f}")
            
            if self.verbose:
                if self.notebook:
                    if self.notebook:
                        this_time = time.time()
                        iter_speed = 1 / (this_time - last_time)
                        last_time = this_time
                        print(f"\rtime {t:>6.2f}/{self.ttotal:<6.2f} | "
                              f"S: {self.num_suc:>4d}, I:{self.num_inf:>4d}, "
                              f"R: {self.num_rec:d}, H: {self.num_treated:d}, "
                              f"lY: {lambdaY.sum():.2f}, lW: {lambdaW.sum():.2f}, lN: {lambdaN.sum():.2f} | "
                              f"{iter_speed:.2f} iter/s"
                              "           ", 
                              end='')
                else:
                    progress_bar.update(np.round(w, 2))

            # Sample what process the arrival came from
            p = np.hstack((lambdaY, lambdaW, lambdaN)) / lambda_all
            p[p == 0.] = 0.  # sets -0.0 to 0.0
            k = np.random.choice(3 * self.n_nodes, p=p)

            # Adjust state variables accordingly
            if k < self.n_nodes:  # arrival Y (infection)
                self.last_arrival_type = 'Y'
                i = k
                
                if self.debug:
                    print(f'Process INFECTION event at node {i:d}')

                self.num_suc -= 1
                self.num_inf += 1

                self.Y[i].generate_arrival_at(t)

                # dX = dY - dW
                self.X[i].generate_arrival_at(t, 1.0)

                # dZ = A(dY - dW)
                neighbor_indices = self.A[i].nonzero()[1]
                for j in neighbor_indices:
                    self.Z[j].generate_arrival_at(t, 1.0)

            elif self.n_nodes <= k < 2 * self.n_nodes:  # arrival W (recovery)
                self.last_arrival_type = 'W'
                i = k - self.n_nodes

                if self.debug:
                    print(f'Process RECOVERY event at node {i:d}')

                self.num_inf -= 1
                self.num_rec += 1

                self.W[i].generate_arrival_at(t)

                # dX = dY - dW
                self.X[i].generate_arrival_at(t, -1.0)

                neighbor_indices = self.A[i].nonzero()[1]
                
                # dZ = A(dY - dW)
                for j in neighbor_indices:
                    self.Z[j].generate_arrival_at(t, -1.0)

                prev_H = self.H[i].value_at(t)
                if prev_H == 1:
                    # dH = dN - H.dW
                    self.H[i].generate_arrival_at(t, -1.0)
                    # dM = A(dN - H.dW)
                    for j in neighbor_indices:
                        self.M[j].generate_arrival_at(t, -1.0)
                    self.num_treated -= 1

            else:  # arrival N (treatment)
                self.last_arrival_type = 'N'
                i = k - 2 * self.n_nodes

                if self.debug:
                    print(f'Process TREATMENT event at node {i:d}')

                self.num_treated += 1

                self.N[i].generate_arrival_at(t)

                # dH = dN - H.dW
                self.H[i].generate_arrival_at(t, 1.0)

                neighbor_indices = self.A[i].nonzero()[1]

                # dM = A(dN - H.dW)
                for j in neighbor_indices:
                    self.M[j].generate_arrival_at(t, 1.0)

            if plot:
                callback(t)

        if plot:
            callback(t, final=True)
            plt.ioff()
            plt.show()

        if self.verbose:
            if self.notebook:
                print()
            else:
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
                                            bounds=(self.min_d0, None),
                                            A_eq=A_eq, b_eq=b_eq, 
                                            options={'tol': 1e-8})
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
