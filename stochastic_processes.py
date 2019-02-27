import bisect


class StochasticProcess:
    """
    General class that handles values of stochastic processes over time.

    Attributes
    ----------
    arrival_times : list
        List of arrival times of the process
    last_arrival : float
        Time of the last arrival
    values : list
        List of values taken by the process at each arrival
    """

    def __init__(self, initial_condition=0.0):
        self.last_arrival = -1.0
        self.arrival_times = []
        self.values = [initial_condition]

    def get_last_arrival_time(self):
        """
        Return the time of the last arrival; None if no arrival happened yet.
        """
        return self.last_arrival if self.last_arrival >= 0.0 else None

    def get_current_value(self):
        """Return current value of stochastic process."""
        return self.values[-1]

    def get_arrival_times(self):
        """Return the list of arrival times."""
        return self.arrival_times

    def generate_arrival_at(self, t, dN=1.0, N=None):
        """Generate an arrival at time t with dN or set to value N if given."""
        # Check that arrival time happens in the present
        if t < self.last_arrival:
            raise ValueError((f"The provided arrival time t=`{t}` is prior to "
                              f"the last recorded arrival time "
                              f"`{self.last_arrival}`"))
        self.last_arrival = t
        self.arrival_times.append(t)
        if N is None:
            self.values.append(self.values[-1] + dN)
        else:
            self.values.append(N)

    def value_at(self, t):
        """Return the value of the stochastic process at time `t`."""
        j = bisect.bisect_right(self.arrival_times, t)
        return self.values[j]


class CountingProcess(StochasticProcess):
    """
    General class that handles values of a counting processes over time.
    A counting process is a particular type of stochastic process that is
    initialized at zero (i.e. `N(0)=0`) and has unit increments (i.e. `dN=1`).
    """

    def __init__(self):
        super().__init__(initial_condition=0.0)

    def generate_arrival_at(self, t):
        """
        Generate the counting process arrival
        """
        super().generate_arrival_at(t, 1.0)


if __name__ == '__main__':
    """Do some basic unit testing."""

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
