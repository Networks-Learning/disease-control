# disease-control

This is the code for the simulation of the project on disease control with SDEs with jumps.
The following describes the different files and how they interact.

# main_simulate.py

This is the main file to run simulations. 
The list of dictionaries named 'settings' contains the settings for which the system will be simulated for, for 'trials_per_setting' trails.
The results are stored as a .pkl pickled file s.t. analysis can be performed later. Make sure to specifiy the file names as desired to know what name the data is stored under.

# main_evaluate.py

The 'names' list of lists contains the names of each of the policies performed in each trial imported.
The 'saved' dictionary specifies a mapping from an integer to a filename and names from 'names', s.t. the 'all_selected' list corresponds to all saved trials that ought to be analyzed. The for loop specifies what evaluations should be performed.

# dynamics.py

This file implements the class 'SISDynamicalSystem' which simulates the dynamical system specified the model under a given policy for a time window, returning the results.
Initialize with the desired parameters.
The ' _simulate' is the core of the class, simulating the arrivals of the counting processes and updating the state variables accordingly. The argument 'policy_fun' is a function specifying the desired control policy. It needs to take a time t and return an array of treatment intensities over all nodes. Several other functions call _simulate to have shorthand calls for different policies, e.g. 'simulate_opt' or 'simulate_trivial' which simulate the system under the stochastic optimal control intensities or trivial control intensities, respectively. 
The function '_getPoissonIntensities' returns the lambdas for Y, W, and N as defined by the model.
' __getOptPolicy(self, t)' implements the optimal policy at time t.

# stochastic_processes.py

Implements the class 'StochasticProcess', and 'CountingProcess' derived from it. The class is designed to keep track of arrival times and value of the stochastic process over time in a convenient way. The comments should be self explanatory.

# analysis.py

This file implements the class 'Evaluation', made to evaluate .pkl result files of main_simulate.py. Hence, main_evaluate.py instantiates an object of this class. Several functions are implemented to create plots of different metrics.
The function '__integrateF' computes integral from 0 to T of e^(eta * t) * f_of_t * dt for a given trial, where f_of_t is tuple returned by the helper function step_sps_values_over_time. The integration is performed by using the fact that all integrals computed in this simulation are piece-wise constant over time windows between arrivals. Thus, the integrals over constant valued time-windows are computed and summed up.

# helpers.py

The class 'HelperFunc' impelements helper functions for dynamics.py and analysis.py. In particular, the extraction of values and arrays from lists of objects of the class 'StochasticProcess'.
