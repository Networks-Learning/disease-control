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


class PriorityQueue(object):

    def __init__(self, list=[], priority=None):

        if priority is None:
            self.priority = lambda x: x
        else:
            self.priority = priority

        self.heap = [(self.priority(x), x) for x in list]
        heapq.heapify(self.heap)

        self.s = len(self.heap)

    def push(self, item):
        heapq.heappush(self.heap, (self.priority(item), item))
        self.s += 1

    def pop(self):
        self.s -= 1
        return heapq.heappop(self.heap)[1]

    def peak(self):
        return self.heap[0][1]

    def size(self):
        return self.s

    def __str__(self):
        return str(self.heap)

    def __repr__(self):
        return repr(self.heap)


class OrderedProcessingList(object):
    """
    List of ('event','time') ordered by 'time' used for the cascades
    'time' is assumed to be a float
    The data structure is implemented using a priority queue
    """

    def __init__(self):
        self.list = PriorityQueue(list=[], priority=lambda x: x[1])

    def __setitem__(self, event, time):
        self.list.push((event, time))

    def pop(self, index):
        return self.list.pop()

    def __len__(self):
        return self.list.size()

    def __str__(self):
        return str(self.list)

    def __repr__(self):
        return repr(self.list)


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
    Attributes:
    ----------
    G : networkx.Graph
        Propagation network, the ids of the nodes must be consecutive integers
        starting at 0.
    n_nodes : int
        Number of nodes in the graph G
    beta : float
        Exponential infection rate (non-negative)
    gamma : float
        Exponential recovery rate (non-negative)
    status : numpy.ndarray
        Array of shape `n_nodes` indicating the status of each node,
        `0` stands for susceptible or healthy,
        `1` stands for infected, 
        `2` stands for recovered or dead
    inf_time : numpy.ndarray
        Array of shape `n_nodes` indicating the time of infection of each node,
        The default value for each not-infected nodes is infinity
    rec_time : numpy.ndarray
        Array of shape `n_nodes` indicating the time of recovery of each node,
        The default value for each not-recovered nodes is infinity
    infector :  numpy.ndarray
        Array of shape `n_nodes` indicating the who infected who
        The default value for each not-infected nodes is NaN
    """
    STATE_SPACE = [0, 1, 2]

    def __init__(self, G, beta, gamma, verbose=True):
        """
        Init an SIR cascade over a graph
        Arguments:
        ---------
        G : networkx.Graph()
                Graph over which the epidemic propagates
        beta : float
            Exponential infection rate (must be non-negative)
        gamma : float
            Exponential recovery rate (must be non-negative)
        verbose : bool (default: True)
            Indicate the print behavior, if set to False, nothing will be
            printed
        """
        # Propagatin network
        if not isinstance(G, nx.Graph):
            raise ValueError('Invalid graph type, must be networkx.Graph')
        # if not set(G.nodes()) == set(range(len(G.nodes()))):
        #     raise ValueError('Invalid node ordering')
        self.G = G
        # Cache the number of nodes
        self.n_nodes = len(G.nodes())
        self.idx_to_node = dict(zip(range(self.n_nodes), G.nodes()))
        self.node_to_idx = dict(zip(G.nodes(), range(self.n_nodes)))
        # Infection rate
        if beta < 0:
            raise ValueError('Invalid `beta` {} (must be non-negative)')
        self.beta = beta
        # Recovery rate
        if gamma < 0:
            raise ValueError('Invalid `gamma` {} (must be non-negative)')
        self.gamma = gamma
        # Printer for logging
        self._printer = ProgressPrinter(verbose=verbose)

    def get_node_status(self, node, time):
        """
        Get the status of a node at a given time
        """
        try:
            if self.inf_time[node] > time:
                return 0
            elif self.rec_time[node] > time:
                return 1
            else:
                return 2
        except IndexError:
            raise ValueError('Invalid node `{}`'.format(node))

    def _draw_edge_delay(self):
        """
        Draw the infection delay of every edge
        """
        edge_list = self.G.edges()
        n_edges = len(edge_list)
        edge_delay = rd.exponential(1./self.beta, size=n_edges)
        self._edge_delay = {}
        for (u, v), d in zip(edge_list, edge_delay):
            self._edge_delay[(u, v)] = d
            self._edge_delay[(v, u)] = d

    def _draw_node_delay(self):
        """
        Draw the recovery delay of every node
        """
        node_list = self.G.nodes()
        n_nodes = len(node_list)
        node_delay = rd.exponential(1./self.gamma, size=n_nodes)
        self._node_delay = {}
        for n, d in zip(node_list, node_delay):
            self._node_delay[n] = d

    def _process_child_infection(self, node, recovery_time, child, time):
        """Deal with neighbors infections"""
        infection_time = time + self._edge_delay[(node, child)]
        if infection_time <= self.max_time:
            if infection_time < recovery_time:
                child_idx = self.node_to_idx[child]
                if self.inf_time[child_idx] > infection_time:
                    self.inf_time[child_idx] = infection_time
                    self.infector[child_idx] = node
                    self.processing[(child, 'inf', True)] = infection_time

    def _process_infection_event(self, node, time, is_normal):
        """
        Mark node `node` as infected at time `time`, then set its recovery 
        time and neighbors infection times to the processing list
        """
        node_idx = self.node_to_idx[node]
        # IF node is already infected do nothing
        if self.status[node_idx] != 0:
            return
        # Mark node as infected
        self.status[node_idx] = 1

        this_infector = self.infector[node_idx]
        if ~np.isnan(this_infector):
            this_infector_idx = self.node_to_idx[this_infector]
            self.num_child_inf[this_infector_idx] += 1

        # Set infection time
        assert (self.inf_time[node_idx] == np.inf) or (
            self.inf_time[node_idx] == time)
        self.inf_time[node_idx] = time
        recovery_time = time + self._node_delay[node]
        if is_normal:  # If the event is from an artificial seed, it is not normal
            # Set recovery event only
            self.processing[(node, 'rec', True)] = recovery_time
        # Set neighbors infection events
        for child in self.G.neighbors(node):
            child_idx = self.node_to_idx[child]
            if self.status[child_idx] == 0:
                self._process_child_infection(node, recovery_time, child, time)

    def _process_recovery_event(self, node, time):
        """
        Mark node `node` as recovered at time `time`
        """
        if time <= self.max_time:
            node_idx = self.node_to_idx[node]
            self.rec_time[node_idx] = time
            self.status[node_idx] = 2

    def _init_run(self, init_event_list, max_time):
        """
        Initialize the run of the epidemic
        """
        # Edge delays for infections
        self._draw_edge_delay()
        # Node delays for recoveries
        self._draw_node_delay()
        # Nodes status (0: Susceptible, 1: Infected, 2: Recovered)
        self.status = np.zeros(self.n_nodes, dtype='int8')
        # Infection times (inf by default)
        self.inf_time = np.inf * np.ones(self.n_nodes, dtype='float')
        # Keep track of who infected who (nan by default)
        self.infector = np.nan * np.ones(self.n_nodes, dtype='int')
        self.num_child_inf = np.zeros(self.n_nodes, dtype='int')
        # Recovery times (inf by default)
        self.rec_time = np.inf * np.ones(self.n_nodes, dtype='float')
        # Maximum epidemic time
        self.max_time = max_time
        # Events to process in order
        self.processing = OrderedProcessingList()
        # Add the initial events
        for event, time in init_event_list:
            source, event_type, _ = event
            source_idx = self.node_to_idx[source]
            if event_type == 'inf':
                self.inf_time[source_idx] = time
                self.infector[source_idx] = np.nan
                self.processing[event] = time
            elif event_type == 'rec':
                self.rec_time[source_idx] = time
                self._node_delay[source] = time - self.inf_time[source_idx]
                self.processing[event] = time
            else:
                raise ValueError('Invalid Event Type')

    def launch_epidemic(self, init_event_list, max_time=np.inf):
        """
        Run the epidemic, starting from node 'source', for at most `max_time` 
        units of time
        """
        self._init_run(init_event_list, max_time)
        # Init epidemic time to 0
        time = 0
        last_print = 0
        while self.processing:
            # Get the next event to process
            (node, event_type, is_normal), time = self.processing.pop(0)
            if time > self.max_time:
                time = self.max_time
                break  # Stop at then end of the observation window
            # Process the event
            if event_type == 'inf':
                # print(f'Process {event_type} ({is_normal}) of node {node} at time {time:.4f} (infector: {self.infector[self.node_to_idx[node]]})')
                self._process_infection_event(node, time, is_normal)
            elif event_type == 'rec':
                # print(f'Process {event_type} ({is_normal}) of node {node} at time {time:.4f}')
                self._process_recovery_event(node, time)
            else:
                raise ValueError("Invalid event type")
            # print(self.status)
            self._printer.print(self, time)
        self._printer.println(self, time)
        # Free memory
        del self.processing
