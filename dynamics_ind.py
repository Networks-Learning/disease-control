"""
epidemic_helper.py: Helper module to simulate continuous-time stochastic 
SIR epidemics.
Copyright © 2018 — LCA 4
"""
import time
import bisect
import numpy as np
import networkx as nx
from numpy import random as rd
import heapq
import collections, itertools


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
    _PRINT_MSG = ('Epidemic spreading... '
                  '{t:.2f} days elapsed | '
                  '{S:.1f}% susceptible, {I:.1f}% infected, '
                  '{R:.1f}% recovered')
    _PRINTLN_MSG = ('Epidemic stopped after {t:.2f} days | '
                    '{t:.2f} days elapsed | '
                    '{S:.1f}% susceptible, {I:.1f}% infected, '
                    '{R:.1f}% recovered')

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.last_print = time.time()

    def print(self, sir_obj, epitime, end='', force=False):
        if not self.verbose:
            return
        if (time.time() - self.last_print > self.PRINT_INTERVAL) or force:
            S = np.sum(sir_obj.status == 0) * 100. / sir_obj.n_nodes
            I = np.sum(sir_obj.status == 1) * 100. / sir_obj.n_nodes
            R = np.sum(sir_obj.status == 2) * 100. / sir_obj.n_nodes
            print('\r', self._PRINT_MSG.format(t=epitime, S=S, I=I, R=R),
                  sep='', end=end, flush=True)
            self.last_print = time.time()

    def println(self, sir_obj, epitime):
        if not self.verbose:
            return
        S = np.sum(sir_obj.status == 0) * 100. / sir_obj.n_nodes
        I = np.sum(sir_obj.status == 1) * 100. / sir_obj.n_nodes
        R = np.sum(sir_obj.status == 2) * 100. / sir_obj.n_nodes
        print('\r', self._PRINTLN_MSG.format(t=epitime, S=S, I=I, R=R),
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
    STATE_SPACE = [0, 1, 2]

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
        
        
        if not isinstance(G, nx.Graph):
            raise ValueError('Invalid graph type, must be networkx.Graph')
        self.G = G

        # Cache the number of nodes
        self.n_nodes = len(G.nodes())
        self.idx_to_node = dict(zip(range(self.n_nodes), G.nodes()))
        self.node_to_idx = dict(zip(G.nodes(), range(self.n_nodes)))

        # Check parameters
        self.beta = param_dict['beta']
        self.gamma = param_dict['gamma']
        self.delta = param_dict['delta']
        self.rho = param_dict['rho']
        self.eta = param_dict['eta']
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

        # Printer for logging
        self._printer = ProgressPrinter(verbose=verbose)

    def expo(self, rate):
        'Samples a single exponential random variable.'
        return rd.exponential(scale=1.0/rate)

    def get_node_status(self, u, time):
        """
        Get the status of a node at a given time
        """
        u_idx = self.node_to_idx[u]
        try:
            if self.inf_occured_at[u_idx] > time:
                return 0
            elif self.rec_occured_at[u_idx] > time:
                return 1
            else:
                return 2
        except IndexError:
            raise ValueError('Invalid node `{}`'.format(node))

    def _init_run(self, init_event_list, max_time):
        """
        Initialize the run of the epidemic
        """

        self.max_time = max_time

        # Priority queue of events by time 
        # event invariant is ('node', event, 'node') where the second node is the infector if applicable
        self.queue = PriorityQueue()
    
        # Node status (0: Susceptible, 1: Infected, 2: Recovered)
        self.initial_seed = np.zeros(self.n_nodes, dtype='bool')

        # Infection tracking
        self.inf_occured_at = np.inf * np.ones(self.n_nodes, dtype='float')      # time infection of u occurred
        self.is_inf = np.zeros(self.n_nodes, dtype='bool')                       # True if u infected
        self.sched_inf_time = np.inf * \
            np.ones((self.n_nodes, self.n_nodes), dtype='float')                 # planned infection time over edge u,v
        self.inf_valid = np.zeros((self.n_nodes, self.n_nodes), dtype='bool')    # True if infection over u,v is valid
        self.infector = np.nan * np.ones(self.n_nodes, dtype='int')              # node that infected u
        self.num_child_inf = np.zeros(self.n_nodes, dtype='int')                 # number of neighbors u infected
        
        # Recovery tracking
        self.rec_occured_at = np.inf * np.ones(self.n_nodes, dtype='float')      # time recovery of u occured
        self.is_rec = np.zeros(self.n_nodes, dtype='bool')                       # True if u recovered
        self.sched_rec_time = np.inf * np.ones(self.n_nodes, dtype='float')      # planned recovery time of u
        
        # Treatment tracking
        self.tre_occured_at = np.inf * np.ones(self.n_nodes, dtype='float')      # time treatment of u occured
        self.is_tre = np.zeros(self.n_nodes, dtype='bool')                       # True if u treated
        self.sched_tre_time = np.inf * np.ones(self.n_nodes, dtype='float')      # planned treatment time of u

        # Add the initial events to priority queue
        for event, time in init_event_list:
            u, event_type, _ = event
            u_idx = self.node_to_idx[u]
            self.initial_seed[u_idx] = True

            if event_type == 'inf':
                # Initial infections have infections from u to u
                self.sched_inf_time[u_idx, u_idx] = time
                self.inf_valid[u_idx, u_idx] = True
                self.queue.push(event, priority=time)

            elif event_type == 'rec':
                self.sched_rec_time[u_idx] = time
                self.queue.push(event, priority=time)
            else:
                raise ValueError('Invalid Event Type for initial seeds.')
           
    def launch_epidemic(self, init_event_list, max_time=np.inf):
        """
        Run the epidemic, starting from initial event list, for at most `max_time` 
        units of time
        """

        self._init_run(init_event_list, max_time)
        time = 0.0
        
        while self.queue:
            # Get the next event to process
            (u, event_type, w), time = self.queue.pop_priority()
            u_idx = self.node_to_idx[u]            

            # Stop at then end of the observation window
            if time > self.max_time:
                time = self.max_time
                break  

            # Process the event
            if (event_type == 'inf') and (not self.is_inf[u_idx]):
                # Check validity of infection event 
                w_idx = self.node_to_idx[w]
                if self.initial_seed[u_idx] or (self.inf_valid[w_idx, u_idx]):
                    self._process_infection_event(u, time, w)
            elif (event_type == 'rec') and (not self.is_rec[u_idx]):
                self._process_recovery_event(u, time)
            elif (event_type == 'tre') and (not self.is_tre[u_idx]):
                self._process_treatment_event(u, time)

            # Update Control
            # TODO

            self._printer.print(self, time)
        self._printer.println(self, time)
        
        # Free memory
        del self.queue

    def _process_infection_event(self, u, time, w):
        """
        Mark node `u` as infected at time `time`
        Sample its recovery time and its neighbors infection times and add to the queue
        """

        u_idx = self.node_to_idx[u]

        # Handle infection
        self.is_inf[u_idx] = True
        self.inf_occured_at[u_idx] = time
        
        if not self.initial_seed[u_idx]:
            w_idx = self.node_to_idx[w]
            self.infector[u_idx] = w
            self.num_child_inf[w_idx] += 1
            recovery_time_u = time + self.expo(self.delta)
            self.sched_rec_time[u_idx] = recovery_time_u
            self.queue.push((u, 'rec', None), priority=recovery_time_u)

        else:
            # Handle initial seeds
            self.infector[u_idx] = np.nan
            recovery_time_u = self.sched_rec_time[u_idx]

        # Set neighbors infection events
        for v in self.G.neighbors(u):
            if self.get_node_status(v, time) == 0:
                v_idx = self.node_to_idx[v]
                infection_time_v = time + self.expo(self.beta)
                if infection_time_v <= self.max_time:
                    self.queue.push((v, 'inf', u), priority=infection_time_v)
                    self.sched_inf_time[u_idx, v_idx] = infection_time_v
                    self.inf_valid[u_idx, v_idx] = (
                        infection_time_v < recovery_time_u)
                            
    def _process_recovery_event(self, u, time):
        """
        Mark node `node` as recovered at time `time`
        """
        u_idx = self.node_to_idx[u]
        self.rec_occured_at[u_idx] = time
        self.is_rec[u_idx] = True

    def _process_treatment_event(self, u, time):
        """
        Mark node `u` as treated at time `time`
        Update its recovery time and its neighbors infection times and the queue
        """
        u_idx = self.node_to_idx[u]

        self.tre_occured_at[u_idx] = time
        self.is_tre[u_idx] = True

        # Update own recovery event with rejection sampling
        assert(self.rho < 0)
        if np.random.uniform() < - self.rho / self.delta:
            # reject previous event
            self.queue.delete((u, 'rec', None))

            # re-sample 
            new_recovery_time_u = time + self.expo(self.delta + self.rho)
            self.queue.push((u, 'rec', None), priority=new_recovery_time_u)
            self.sched_rec_time[u_idx] = new_recovery_time_u

        # Update neighbors infection events triggered by u 
        for v in self.G.neighbors(u):
            if self.get_node_status(v, time) == 0:
                v_idx = self.node_to_idx[v]

                if np.random.uniform() < self.gamma / self.beta:

                    # reject previous event
                    self.queue.delete((v, 'inf', u))

                    # re-sample
                    new_infection_time_v = time + self.expo(self.beta - self.gamma)
                    self.queue.push((v, 'inf', u), priority=new_infection_time_v)
                    self.sched_inf_time[u_idx, v_idx] = new_infection_time_v

                # check validity of infection from u to v with new times
                self.inf_valid[u_idx, v_idx] = (
                    self.sched_inf_time[u_idx, v_idx] < self.sched_rec_time[u_idx])



