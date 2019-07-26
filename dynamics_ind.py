"""
epidemic_helper.py: Helper module to simulate continuous-time stochastic 
SIR epidemics.
Copyright © 2018 — LCA 4
"""
import time
import bisect
import numpy as np
import networkx as nx
import scipy 
import scipy as sp
from numpy import random as rd
import heapq
import collections, itertools
import maxcut
from lpsolvers import solve_lp


class PriorityQueue(object):

    """
    PriorityQueue with O(1) update and deletion of objects
    """

    def __init__(self, initial=[], priorities=[]):

        self.pq = []
        self.entry_finder = {}               # mapping of tasks to entries
        self.REMOVED = '<removed-task>'      # placeholder for a removed task
        self.counter = itertools.count()     # unique sequence count

        assert(len(initial) == len(priorities))
        for i in range(len(initial)):
            self.push(initial[i], priority=priorities[i])

    def push(self, task, priority=0):
        'Add a new task or update the priority of an existing task'
        if task in self.entry_finder:
            self.delete(task)
        count = next(self.counter)
        entry = [priority, count, task]
        self.entry_finder[task] = entry
        heapq.heappush(self.pq, entry)

    def delete(self, task):
        'Mark an existing task as REMOVED.  Raise KeyError if not found.'
        entry = self.entry_finder.pop(task)
        entry[-1] = self.REMOVED

    def remove_all_tasks_of_type(self, type):
        'Removes all existing tasks of a specific type (for SIRSimulation)'
        keys = list(self.entry_finder.keys())
        for event in keys:
            u, type_, v = event
            if type_ == type:
                self.delete(event)

    def pop_priority(self):
        'Remove and return the lowest priority task with its priority value.'
        'Raise KeyError if empty.'
        while self.pq:
            priority, _, task = heapq.heappop(self.pq)
            if task is not self.REMOVED:
                del self.entry_finder[task]
                return task, priority
        raise KeyError('pop from an empty priority queue')

    def pop(self):
        'Remove and return the lowest priority task. Raise KeyError if empty.'
        task, _ = self.pop_priority()
        return task

    def priority(self, task):
        'Return priority of task'
        if task in self.entry_finder:
            return self.entry_finder[task][0]
        else:
            raise KeyError('task not in queue')
        
    def __len__(self):
        return len(self.entry_finder)

    def __str__(self):
        return str(self.pq)

    def __repr__(self):
        return repr(self.pq)

    def __setitem__(self, task, priority):
        self.push(task, priority=priority)


class ProgressPrinter(object):
    """
    Helper object to print relevant information throughout the epidemic
    """
    PRINT_INTERVAL = 0.1
    _PRINT_MSG = ('{t:.2f} days elapsed | '
                  '{S:.0f} sus., '
                  '{I:.0f} inf., '
                  '{R:.0f} rec. | '
                  '{Tt:.0f} treated ({TI:.2f}% of infected) | I(q): {iq} R(q): {rq}')
    _PRINTLN_MSG = ('Epidemic stopped after {t:.2f} days | '
                    '{S:.0f} sus., '
                    '{I:.0f} inf., '
                    '{R:.0f} rec. | '
                    '{Tt:.0f} treated ({TI:.2f}% of infected)')

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.last_print = time.time()

    def print(self, sir_obj, epitime, end='', force=False):
        if not self.verbose:
            return
        if (time.time() - self.last_print > self.PRINT_INTERVAL) or force:
            S = np.sum(sir_obj.is_sus) 
            I = np.sum(sir_obj.is_inf) 
            T = np.sum(sir_obj.is_tre) 
            R = np.sum(sir_obj.is_rec)
            Tt = np.sum(sir_obj.is_tre)
            TI = 100. * T / I
            

            iq = sir_obj.infs_in_queue
            rq = sir_obj.recs_in_queue
           
            print('\r', self._PRINT_MSG.format(
                t=epitime, S=S, I=I, T=T, Tt=Tt, R=R, TI=TI, iq=iq, rq=rq), #q=len(sir_obj.queue.pq), ef=len(sir_obj.queue.entry_finder)),
                sep='', end='', flush=True)
            self.last_print = time.time()

    def println(self, sir_obj, epitime):
        if not self.verbose:
            return
        S = np.sum(sir_obj.is_sus) #* 100. / sir_obj.n_nodes
        I = np.sum(sir_obj.is_inf) #* 100. / sir_obj.n_nodes
        T = np.sum(sir_obj.is_tre) #* 100. / sir_obj.n_nodes
        Tt = np.sum(sir_obj.is_tre) 
        TI = 100. * T / I
        R = np.sum(sir_obj.is_rec) * 100. / sir_obj.n_nodes
        print('\r', self._PRINTLN_MSG.format(t=epitime, S=S, I=I, T=T, Tt=Tt, R=R, TI=TI),
              sep='', end='\n', flush=True)
        self.last_print = time.time()


class SimulationSIR(object):
    """
    Simulate continuous-time SIR epidemic with exponentially distributed
    infection and recovery rates.

    Invariant of an event in the queue is 
        (node, event_str, infector node)

    Attributes:
    ----------
    G : networkx.Graph
        Propagation network

    beta : float
        Exponential infection rate (non-negative)
    gamma : float
        Reduction in infection rate by treatment 
    delta : float
        Exponential recovery rate (non-negative)
    rho : float
        Increase in recovery rate by treatment

    """
    

    def __init__(self, G, param_dict, verbose=True):
        """
        Init an SIR cascade over a graph
        Arguments:
        ---------
        G : networkx.Graph()
                Graph over which the epidemic propagates
        param_dict : dict
            Dict with all epidemic parameters
        verbose : bool (default: True)
            Indicate the print behavior, if set to False, nothing will be
            printed
        """
        
        self.LPSOLVER = ['scipy', 'cvxopt'][1]

        if not isinstance(G, nx.Graph):
            raise ValueError('Invalid graph type, must be networkx.Graph')
        self.G = G
        self.A = sp.sparse.csr_matrix(nx.adjacency_matrix(self.G).toarray())

        # Cache the number of nodes
        self.n_nodes = len(G.nodes())
        self.max_deg = np.max([d for n, d in self.G.degree()])
        self.min_deg = np.min([d for n, d in self.G.degree()])
        self.idx_to_node = dict(zip(range(self.n_nodes), self.G.nodes()))
        self.node_to_idx = dict(zip(self.G.nodes(), range(self.n_nodes)))

        # Check parameters
        self.beta = param_dict['beta']
        self.gamma = param_dict['gamma']
        self.delta = param_dict['delta']
        self.rho = param_dict['rho']
        self.eta = param_dict['eta']
        self.q_x = param_dict['q_x']
        self.q_lam = param_dict['q_lam']

        if self.beta < 0:
            raise ValueError('Invalid `beta` (must be non-negative)')
        if self.gamma < 0 or self.gamma > self.beta:
            raise ValueError(
                'Invalid `gamma` (must be non-negative) and smaller than `beta`')
        if self.delta < 0:
            raise ValueError('Invalid `delta` (must be non-negative)')
        if self.rho > 0 or self.rho + self.delta < 0:
            raise ValueError(
                'For the Ebola application and this code '
                '`rho` (must be non-positive) and `delta + rho` must be non-negative')

        # Control pre-computations
        self.lrsr_initiated = False   # flag for initial LRSR computation
        self.mcm_initiated = False    # flag for initial MCM computation

        self.stored_matrices = None 

    
        # Printer for logging
        self._printer = ProgressPrinter(verbose=verbose)

    def expo(self, rate):
        'Samples a single exponential random variable.'
        return rd.exponential(scale=1.0/rate)

    def nodes_at_time(self, status, time):
        """
        Get the status of all nodes at a given time
        """
        if status == 'S':
            return self.inf_occured_at > time
        elif status == 'I':
            return (self.rec_occured_at > time) * (self.inf_occured_at < time)
        elif status == 'T':
            return (self.tre_occured_at < time) * (self.rec_occured_at > time)
        elif status == 'R':
            return self.rec_occured_at < time
        else:
            raise ValueError('Invalid status.')

    def _init_run(self, init_event_list, max_time):
        """
        Initialize the run of the epidemic
        """

        self.infs_in_queue = 0
        self.recs_in_queue = 0


        self.max_time = max_time

        # Priority queue of events by time 
        # event invariant is ('node', event, 'node') where the second node is the infector if applicable
        self.queue = PriorityQueue()
    
        # Node status (0: Susceptible, 1: Infected, 2: Recovered)
        self.initial_seed = np.zeros(self.n_nodes, dtype='bool')
        self.is_sus = np.ones(self.n_nodes, dtype='bool')                        # True if u susceptible

        # Infection tracking
        self.inf_occured_at = np.inf * np.ones(self.n_nodes, dtype='float')      # time infection of u occurred
        self.is_inf = np.zeros(self.n_nodes, dtype='bool')                       # True if u infected
        self.infector = np.nan * np.ones(self.n_nodes, dtype='int')              # node that infected u
        self.num_child_inf = np.zeros(self.n_nodes, dtype='int')                 # number of neighbors u infected
        
        # Recovery tracking
        self.rec_occured_at = np.inf * np.ones(self.n_nodes, dtype='float')      # time recovery of u occured
        self.is_rec = np.zeros(self.n_nodes, dtype='bool')                       # True if u recovered
        
        # Treatment tracking
        self.tre_occured_at = np.inf * np.ones(self.n_nodes, dtype='float')      # time treatment of u occured
        self.is_tre = np.zeros(self.n_nodes, dtype='bool')                       # True if u treated

        # Conrol tracking
        self.old_lambdas = np.zeros(self.n_nodes, dtype='float')                 # control intensity of prev iter
        self.max_interventions_reached = False

        # Add the initial events to priority queue
        for event, time in init_event_list:
            u, event_type, _ = event
            u_idx = self.node_to_idx[u]
            self.initial_seed[u_idx] = True

            if event_type == 'inf':
                # Initial infections have infections from u to u
                self.queue.push(event, priority=time)

            elif event_type == 'rec':
                self.queue.push(event, priority=time)
            else:
                raise ValueError('Invalid Event Type for initial seeds.')
        
    def _process_infection_event(self, u, time, w):
        """
        Mark node `u` as infected at time `time`
        Sample its recovery time and its neighbors infection times and add to the queue
        """
        u_idx = self.node_to_idx[u]

        # Handle infection
        self.is_inf[u_idx] = True
        self.is_sus[u_idx] = False
        self.inf_occured_at[u_idx] = time
        self.infs_in_queue -= 1

        if not self.initial_seed[u_idx]:
            w_idx = self.node_to_idx[w]
            self.infector[u_idx] = w
            self.num_child_inf[w_idx] += 1
            recovery_time_u = time + self.expo(self.delta)
            self.queue.push((u, 'rec', None), priority=recovery_time_u)
            self.recs_in_queue += 1

        else:
            # Handle initial seeds
            self.infector[u_idx] = np.nan

        # Set neighbors infection events
        for v in self.G.neighbors(u):
            v_idx = self.node_to_idx[v]
            if self.is_sus[v_idx]:
                infection_time_v = time + self.expo(self.beta)
                self.queue.push((v, 'inf', u), priority=infection_time_v)
                self.infs_in_queue += 1
           
    def _process_recovery_event(self, u, time):
        """
        Mark node `node` as recovered at time `time`
        """
        u_idx = self.node_to_idx[u]
        self.rec_occured_at[u_idx] = time
        self.is_rec[u_idx] = True
        self.recs_in_queue -= 1

    def _process_treatment_event(self, u, time):
        """
        Mark node `u` as treated at time `time`
        Update its recovery time and its neighbors infection times and the queue
        """
        u_idx = self.node_to_idx[u]

        self.tre_occured_at[u_idx] = time
        self.is_tre[u_idx] = True

        # Update own recovery event with rejection sampling
        assert(self.rho <= 0)
        if np.random.uniform() < - self.rho / self.delta:
            # reject previous event
            self.queue.delete((u, 'rec', None))

            # re-sample 
            new_recovery_time_u = time + self.expo(self.delta + self.rho)
            self.queue.push((u, 'rec', None), priority=new_recovery_time_u)

        # Update neighbors infection events triggered by u 
        for v in self.G.neighbors(u):
            v_idx = self.node_to_idx[v]
            if self.is_sus[v_idx]:  
                if np.random.uniform() < self.gamma / self.beta:

                    # reject previous event
                    self.queue.delete((v, 'inf', u))

                    # re-sample
                    new_infection_time_v = time + self.expo(self.beta - self.gamma)
                    self.queue.push((v, 'inf', u), priority=new_infection_time_v)

    def _control(self, u, time, policy='NO'):

        u_idx = self.node_to_idx[u]

        # Check if max interventions were reached (for FL)
        if '-FL' in policy:
            max_interventions = self.policy_dict['front-loading']['max_interventions']
            current_interventions = np.sum(self.is_tre)
            if current_interventions > max_interventions:

                # End interventions for this simulation
                self.max_interventions_reached = True
                self.queue.remove_all_tasks_of_type('tre')
                print('All treatments ended')
                return

        # Compute control intensity
        self.new_lambda = self._compute_lambda(u, time, policy=policy)

        # Sample treatment event
        delta = self.new_lambda - self.old_lambdas[u_idx]
        if delta < 0:
            # Update treatment event with rejection sampling as intensity was reduced
            if np.random.uniform() < 1 - self.new_lambda / self.old_lambdas[u_idx]:
                # reject previous event
                self.queue.delete((u, 'tre', None))

                if self.new_lambda > 0:
                    # re-sample
                    new_treatment_time_u = time + self.expo(self.new_lambda)
                    self.queue.push((u, 'tre', None), priority=new_treatment_time_u)

        elif delta > 0:
            # Sample new/additional treatment event with the superposition principle
            new_treatment_time_u = time + self.expo(delta)
            self.queue.push((u, 'tre', None), priority=new_treatment_time_u)

        # store lambda
        self.old_lambdas[u_idx] = self.new_lambda 

    def _compute_lambda(self, u, time, policy='NO'):
        'Computes control intensity of the respective policy'

        if policy == 'NO':
            return 0.0

        elif policy == 'TR':
            # lambda = const.
            return self.policy_dict['TR']

        elif policy == 'TR-FL':
            return self._frontloadPolicy(
                self.policy_dict['TR'], 
                self.policy_dict['TR'], time)

        elif policy == 'MN':
            # lambda ~ deg(u)
            return self.G.degree(u) * self.policy_dict['MN']

        elif policy == 'MN-FL':
            return self._frontloadPolicy(
                self.G.degree(u) * self.policy_dict['MN'], 
                self.max_deg * self.policy_dict['MN'], time)

        elif policy == 'LN':
            # lambda ~ (maxdeg - deg(u) + 1)
            return (self.max_deg - self.G.degree(u) + 1) * self.policy_dict['LN']

        elif policy == 'LN-FL':
            return self._frontloadPolicy(
                (self.max_deg - self.G.degree(u) + 1) * self.policy_dict['LN'],
                (self.max_deg - self.min_deg + 1) * self.policy_dict['LN'], time)

        elif policy == 'LRSR':
            # lambda ~ 1/rank
            # where rank is order of largest reduction in spectral radius of A
            intensity, _ = self._compute_LRSR_lambda(u, time)
            return intensity

        elif policy == 'LRSR-FL':
            intensity, max_intensity = self._compute_LRSR_lambda(u, time)
            return self._frontloadPolicy(
                intensity, max_intensity, time)

        elif policy == 'MCM':
            # lambda ~ 1/rank
            # where rank is MCM heuristic ranking
            intensity, _ = self._compute_MCM_lambda(u, time)
            return intensity

        elif policy == 'MCM-FL':
            intensity, max_intensity = self._compute_MCM_lambda(u, time)
            return self._frontloadPolicy(
                intensity, max_intensity, time)

        elif policy == 'SOC':
            return self._compute_SOC_lambda(u, time)

        else:
            raise KeyError('Invalid policy code.')

    def _frontloadPolicy(self, intensity, max_intensity, time):
        """
        Return front-loaded variation of policy u at time t
        Scales a given `intensity` such that the policy's current
        `max_intensity` is equal to the SOC's `max_lambda`
        """
        max_lambda = self.policy_dict['front-loading']['max_lambda']

        # scale proportionally s.t. max(u) = max(u_SOC)
        if max_intensity > 0.0:
            return max_lambda * intensity / max_intensity
        else:
            return 0.0

    def _compute_LRSR_lambda(self, u, time):
        
        # TODO
        # raise ValueError('Currently too slow for big networks. Eigenvalues of A need to be found |V| times using brute force.')


        # lambda ~ 1/rank
        # where rank is order of largest reduction in spectral radius of A
        if self.lrsr_initiated:
            intensity = 1.0 / (1.0 + np.where(self.spectral_ranking == u)[0]) * self.policy_dict['LRSR']
            max_intensity = self.policy_dict['LRSR']

            # return both u's intensity and max intensity of all nodes for potential FL
            return intensity, max_intensity
        else:
            # first time: compute ranking for all nodes

            def spectral_radius(A): # TODO not tested yet
                return np.max(scipy.linalg.eigvalsh(self.A, turbo=True, eigvals=(self.n_nodes - 2, self.n_nodes - 1)))

            # Brute force:
            # find which node removals reduce spectral radius the most
            tau = spectral_radius(A)
            reduction_by_node = np.zeros(self.n_nodes)
            
            last_print = time.time()
            for n in range(self.n_nodes):
                A_ = np.copy(A)
                A_[n, :] = np.zeros(self.n_nodes)
                A_[:, n] = np.zeros(self.n_nodes)
                reduction_by_node[n] = tau - spectral_radius(A_)

                # printing
                print(100 * n / self.n_nodes)
                if (time.time() - last_print > 0.1):
                    last_print = time.time()
                    done = 100 * n  /self.n_nodes

                    print('\r', f'Computing LRSR ranking... {done:.2f}%',
                        sep='', end='', flush=True)

            order = np.argsort(reduction_by_node)
            self.spectral_ranking_idx = np.flip(order)
            self.spectral_ranking = np.vectorize(self.idx_to_node.get)(self.spectral_ranking_idx)
            self.lrsr_initiated = True

            intensity = 1.0 / (1.0 + np.where(self.spectral_ranking == u)) * self.policy_dict['LRSR']
            max_intensity = self.policy_dict['LRSR']
            
            # return both u's intensity and max intensity of all nodes for potential FL
            return intensity, max_intensity

    def _compute_MCM_lambda(self, u, time):
        """
        Return the adapted heuristic policy MaxCutMinimzation (MCM) at
        time `t`. The method is adapted to fit the setup where treatment
        intensity `rho` is the equal for everyone, and the control is made on
        the rate of intervention, not the intensity of the treatment itself.
        """

        # # TODO
        if self.n_nodes > 5000:
            raise ValueError('Currently too slow for big networks. Eigenvalues of A needed.')

        if self.mcm_initiated:
            intensity = 1.0 / (1.0 + np.where(self.mcm_ranking == u)[0]) * self.policy_dict['MCM']
            max_intensity = self.policy_dict['MCM']
            
            # return both u's intensity and max intensity of all nodes for potential FL
            return intensity, max_intensity

        else:
            # first time: compute ranking for all nodes 
            order = maxcut.mcm(self.A)
            self.mcm_ranking_idx = np.flip(order)

            self.mcm_ranking = np.vectorize(self.idx_to_node.get)(self.mcm_ranking_idx)
            self.mcm_initiated = True

            intensity = 1.0 / (1.0 + np.where(self.mcm_ranking == u)[0]) * self.policy_dict['MCM']
            max_intensity = self.policy_dict['MCM']
            
            # return both u's intensity and max intensity of all nodes for potential FL
            return intensity, max_intensity
            
    def _compute_SOC_lambda(self, u, time):
        'Stochastic optimal control policy'
        
        d = np.zeros(self.n_nodes)
        d[self.lp_d_S_idx] = self.lp_d_S

        K1 = self.beta * (2 * self.delta + self.eta + self.rho)
        K2 = self.beta * (self.delta + self.eta) * (self.delta + self.eta + self.rho) * self.q_lam       
        K3 = self.eta * (self.gamma * (self.delta + self.eta) + self.beta * (self.delta + self.rho))
        K4 = self.beta * (self.delta + self.rho) * self.q_x

        cache = float(np.dot(self.A[self.node_to_idx[u]].toarray(), d))
        intensity = - 1.0 / (K1 * self.q_lam) * (K2 - np.sqrt(2.0 * K1 * self.q_lam * (K3 * cache + K4) + K2 ** 2.0))

        if intensity < 0.0:
            raise ValueError('Control intensity has to be non-negative.')

        return intensity

    def _update_LP_sol(self):
        
        # find subarrays
        x_S = np.where(self.is_sus)[0]
        x_I = np.where(self.is_inf)[0]
        len_S = x_S.shape[0]
        len_I = x_I.shape[0]
        A_IS = self.A[np.ix_(x_I, x_S)]

        K3 = self.eta * (self.gamma * (self.delta + self.eta) + self.beta * (self.delta + self.rho))
        K4 = self.beta * (self.delta + self.rho) * self.q_x

        # objective: c^T x
        c = np.hstack((np.ones(len_I), np.zeros(len_S)))

        # inequality: Ax <= b
        A_ineq = sp.sparse.hstack(
            [sp.sparse.csr_matrix((len_I, len_I)),  - A_IS])

        A_eq = sp.sparse.hstack(
            [- sp.sparse.eye(len_I),  A_IS])

        A = sp.sparse.vstack([A_ineq, A_eq])

        b = np.hstack(
            [K4 / K3 * np.ones(len_I) - 1e-8, 
             -K4 / K3 * np.ones(len_I) - 1e-8]
        )

        # new
        C_ineq = sp.sparse.vstack([A_ineq, A_eq])
        C_ineq_dense = C_ineq.toarray()
        d_ineq = np.hstack([b_ineq, -b_ineq])

        bounds = tuple([(0.0, None)] * len_I + [(None, None)] * len_S)

        self.stored_matrices = (c, C_ineq, d_ineq)


        if self.LPSOLVER == 'scipy':

            result = scipy.optimize.linprog(
                c, A_ub=A, b_ub=b,
                bounds=bounds,
                options={'tol': 1e-8})

            if result['success']:
                d_S = result['x'][len_I:]
            else:
                raise Exception("LP couldn't be solved.")

        elif self.LPSOLVER == 'cvxopt':

            A_dense = A.toarray()
            res = solve_lp(c, A_dense, b, None, None)
            d_S = res[len_I:]

        else:
            raise KeyError('Invalid LP Solver.')

        # store LP solution
        self.lp_d_S = d_S
        self.lp_d_S_idx = x_S
        
    def launch_epidemic(self, init_event_list, max_time=np.inf, policy='NO', policy_dict={}):
        """
        Run the epidemic, starting from initial event list, for at most `max_time` 
        units of time
        """

        self._init_run(init_event_list, max_time)
        self.policy = policy
        self.policy_dict = policy_dict
        time = 0.0

        while self.queue:
            # Get the next event to process
            (u, event_type, w), time = self.queue.pop_priority()

            u_idx = self.node_to_idx[u]

            # print(np.sum(self.nodes_at_time('I', time)),
            #       np.sum(self.nodes_at_time('T', time)))

            # Stop at then end of the observation window
            if time > self.max_time:
                time = self.max_time
                break

            # Process the event
            if (event_type == 'inf') and (not self.is_inf[u_idx]):
                # Check validity of infection event
                w_idx = self.node_to_idx[w]
                if self.initial_seed[u_idx] or (not self.is_rec[w_idx]):
                    self._process_infection_event(u, time, w)
            elif (event_type == 'rec') and (not self.is_rec[u_idx]):
                self._process_recovery_event(u, time)
            elif (event_type == 'tre') and (not self.is_tre[u_idx]) and (not self.is_rec[u_idx]):
                self._process_treatment_event(u, time)

            # Update Control for nodes still untreated and infected
            if not self.max_interventions_reached:
                controlled_nodes = np.where(self.is_inf * (1 - self.is_tre) * (1 - self.is_rec))[0]

                if self.policy == 'SOC':
                    self._update_LP_sol()

                for u_idx in controlled_nodes:
                    self._control(self.idx_to_node[u_idx], time, policy=self.policy)

            # print
            self._printer.print(self, time)

        self._printer.println(self, time)

        # Free memory
        del self.queue


