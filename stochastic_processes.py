
import bisect


'''
General class that handles values of stochastic processes over time
'''
class StochasticProcess:

    def __init__(self, initial_condition=0.0):
        self.last_arrival = -1.0
        self.arrival_times = []
        self.values = [initial_condition]

    '''Returns time of last arrival; None if no arrival happened yet'''

    def get_last_arrival_time(self):
        return self.last_arrival if self.last_arrival >= 0.0 else None

    '''Returns current value of stochastic process'''

    def get_current_value(self):
        return self.values[-1]

    '''Returns array of arrival times'''

    def get_arrival_times(self):
        return self.arrival_times

    '''Generates arrival at time t with dN or set to value N if given'''

    def generate_arrival_at(self, t, dN=1.0, N=None):
        '''Check that arrival time happens in the present'''
        try:
            assert(t > self.last_arrival)
        except AssertionError:
            print("AssertionError: arrival time is before a previous arrival")
            exit(1)

        self.last_arrival = t
        self.arrival_times.append(t)
        if N is None:
            self.values.append(self.values[-1] + dN)
        else:
            self.values.append(N)

    '''Returns value of the stochastic process at time t'''

    def value_at(self, t):
        j = bisect.bisect_right(self.arrival_times, t)
        return self.values[j]


'''
General class that handles values of counting processes over time
'''


class CountingProcess(StochasticProcess):

    def __init__(self):
        StochasticProcess.__init__(self, initial_condition=0.0)

    '''Generates counting process arrival'''
    def generate_arrival_at(self, t):
        super().generate_arrival_at(t, 1.0)



if __name__ == '__main__':

    '''Basic unit testing'''
    s = StochasticProcess(initial_condition=0.0)

    s.generate_arrival_at(1.0, 1.0)
    s.generate_arrival_at(5.0, 1.0)
    s.generate_arrival_at(10.0, -1.0)

    assert(s.get_arrival_times() == [1.0, 5.0, 10.0])
    assert(s.get_current_value() == 1.0)
    assert(s.get_last_arrival_time() == 10.0)

    tests = zip([-2.0, -1.0, 0.0, 1.0, 3.0, 5.0, 7.0, 10.0, 11.0], 
                [0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0])
    for t, sol in tests:
        assert(sol == s.value_at(t))


    c = CountingProcess()

    c.generate_arrival_at(1.0)  
    c.generate_arrival_at(5.0)

    assert(c.get_arrival_times() == [1.0, 5.0])
    assert(c.get_current_value() == 2.0)
    assert(c.get_last_arrival_time() == 5.0)

    tests2 = zip([-2.0, -1.0, 0.0, 1.0, 3.0, 5.0, 7.0],
                 [0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0])

    for t, sol in tests:
        assert(sol == c.value_at(t))

    print("Unit tests successful.")
