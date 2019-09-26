import numpy as np


class HelperFunc:
    """
    Helper functions
    """

    def __init__(self):
        pass

    def all_arrivals(self, sps):
        """
        Extract all the arrivals times from the stochastic processes in `sps`.

        Parameters
        ----------
        sps : numpy.ndarray
            Array of `StochasticProcess`

        Returns
        -------
        arrival_list : list
            Flat list containing all the arrivals in `sps`
        """
        arrivals = [sp.get_arrival_times() for sp in sps]
        return sorted([arrival for sublist in arrivals for arrival in sublist])

    def sps_values(self, sps, t, summed=False):
        """
        Return the value of all stochastic processes in `sps` at time `t`.

        Parameters
        ----------
        sps : numpy.ndarray
            Array of `StochasticProcess`
        summed : bool, optional (default: False)
            If True, returns total number of sps at time t where
            sp.value_at(t) == 1

        Returns
        -------
        value_arr : np.ndarray
            array in {0,1}^N of sps at time t where sp.value_at(t) == 1
            where every StochasticProcess sp in {0,1}

        """
        vconvert = np.vectorize(lambda sp: sp.value_at(t))
        if summed:
            return np.add.reduce(vconvert(sps))
        else:
            return vconvert(sps)

    def sps_values_over_time(self, sps, summed=False):
        """
        Return the value of all stochastic processes evaluated at the union of
        all arrival times

        Parameters
        ----------
        sps : numpy.ndarray
            Array of `StochasticProcess`
        summed : bool, optional (default: False)
            If True, returns total number of sps where sp.value_at(t) == 1 at
            every arrival time of sps

        Returns
        -------
        sps_list : list
            list of sps where sp.value_at(t) == 1 at every arrival time of
            object sp in sps
        """
        all_t = self.all_arrivals(sps)
        if summed:
            return [self.sps_values(sps, t, summed=True) for t in all_t]
        else:
            return [self.sps_values(sps, t, summed=False) for t in all_t]

    def step_sps_values_over_time(self, sps, summed=False):
        """
        Helper function to compute:
            t = `all_arrivals(sps)`, and
            X = `sps_values_over_time(sps, summed=summed)`)
        augmented to cover constant intervals:
        every two consecutive pairs (t1, t2), (t3, t4),.. in
        t = [t1, t2, t3, t4, ..] returned has a constant values in X
        i.e. X[t1] == X[t2], X[t3] == X[t4], ...

        Parameters
        ----------
        sps : numpy.ndarray
            Array of `StochasticProcess`
        summed : bool, optional (default: False)
            If True, returns total number of sps where sp.value_at(t) == 1 at
            every arrival time of sps

        Returns
        -------
        t : list
            Flat list containing all the arrivals in `sps`
        X : list
            Flat list of values taken the stochastic processes in `sps`
        """
        t_ = self.all_arrivals(sps)
        y_ = self.sps_values_over_time(sps, summed=summed)
        t = [] if len(t_) == 0 else \
            [0.0] + [val for val in t_ for _ in (0, 1)][:2 * len(t_) - 1]
        y = [val for val in y_ for _ in (0, 1)]
        return t, y


if __name__ == '__main__':

    # Basic unit testing
    print("TODO")
