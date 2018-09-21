import numpy as np

'''
Helper functions for dynamics.py and analysis.py
'''

class HelperFunc:

    def __init__(self):
        pass
    
    ''' Conversion between array of objects of StochasticProcess and array of values
        Here: sps is np.array of objects of the StochasticProcess class 
    '''

    '''
    all_arrivals(sps) returns flat list of arrivals times of all objects of StochasticProcess in sps
    '''

    def all_arrivals(self, sps):
        arrivals = [sp.get_arrival_times() for sp in sps]
        return sorted([arrival for sublist in arrivals for arrival in sublist])

    '''
    sps_values(sps, t, summed=False) returns array in {0,1}^N of sps at time t where sp.value_at(t) == 1
    where every StochasticProcess sp in {0,1}
        if summed == True: returns total number of sps at time t where sp.value_at(t) == 1
    '''

    def sps_values(self, sps, t, summed=False):
        vconvert = np.vectorize(lambda sp: sp.value_at(t))
        if summed:
            return np.add.reduce(vconvert(sps))
        else:
            return vconvert(sps)

    '''
    sps_values_over_time(sps, summed=False) returns list of sps where sp.value_at(t) == 1
    at every arrival time of object sp in sps
        if summed == True: returns total number of sps where sp.value_at(t) == 1 at every arrival time of sps
    '''

    def sps_values_over_time(self, sps, summed=False):
        all_t = self.all_arrivals(sps)
        if summed:
            return [self.sps_values(sps, t, summed=True) for t in all_t]
        else:
            return [self.sps_values(sps, t, summed=False) for t in all_t]

    '''
    step_sps_values_over_time(sps, summed=False) returns (t, X) where
    (t,X) is (all_arrivals(sps), sps_values_over_time(sps, summed=summed)) augmented to cover constant intervals:
    every two consecutive pairs (t1, t2), (t3, t4),.. in t = [t1, t2, t3, t4, ..] returned has a constant values in X
    i.e. X[t1] == X[t2], X[t3] == X[t4], ...
    '''

    def step_sps_values_over_time(self, sps, summed=False):
        t_ = self.all_arrivals(sps)
        y_ = self.sps_values_over_time(sps, summed=summed)
        t = [] if len(t_) == 0 else \
            [0.0] + [val for val in t_ for _ in (0, 1)][:2 * len(t_) - 1]
        y = [val for val in y_ for _ in (0, 1)]
        return t, y


if __name__ == '__main__':

    '''Basic unit testing'''

    print("TODO")
    pass