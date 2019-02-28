from dynamics import SISDynamicalSystem


class Experiment:
    """
    Class for easier handling of experiments.
    Directly incorporates running the dynamical system simulation.

    (Certain default values are set here to reduce clutter.)
    """

    def __init__(self, name, policy_list=None, sim_dict=None, param_dict=None,
                 cost_dict=None, baselines_dict=None):
        # Default policies to run
        self.policy_list = [
            'SOC',
            'TR',
            'TR-FL',
            'MN',
            'MN-FL',
            'LN',
            'LN-FL',
            'LRSR',
            'MCM',
        ]
        # Default simulation settings
        self.sim_dict = {
            'total_time': 10.00,
            'trials_per_setting': 10
        }
        # Default model parameters
        self.param_dict = {
            'beta':  6.0,
            'gamma': 5.0,
            'delta': 1.0,
            'rho':   5.0,
            'eta':   1.0
        }
        # Default loss function parameterization
        self.cost_dict = {
            'Qlam': 1.0,
            'Qx': 400.0
        }
        # Default proportional scaling for fair comparison
        self.baselines_dict = {
            'TR': 0.003,
            'MN': 0.0007,
            'LN': 0.0008,
            'LRSR': 22.0,
            'MCM': 1.0,  # TODO
            'FL_info': dict(N=356.1000, max_u=19.5352),
        }
        # Experiment name
        self.name = name
        # Change defaults to given parameters
        self.update(policy_list=policy_list,
                    sim_dict=sim_dict,
                    param_dict=param_dict,
                    cost_dict=cost_dict,
                    baselines_dict=baselines_dict)

    def update(self, policy_list=None, sim_dict=None, param_dict=None,
               cost_dict=None, baselines_dict=None):
        """
        Update dictionaries.
        """
        if policy_list:
            self.policy_list = policy_list
        if sim_dict:
            self.sim_dict = sim_dict
        if param_dict:
            self.param_dict = param_dict
        if cost_dict:
            self.cost_dict = cost_dict
        if baselines_dict:
            self.baselines_dict = baselines_dict

    def run(self, A, X_init):
        """
        Run the experiment and return a summary as a list of dict.
        """
        # Initialize the result object
        result = [{"dat": [], "name": policy} for policy in self.policy_list]
        # Simulate number of times
        for tr in range(self.sim_dict['trials_per_setting']):
            # ...for every requested policy
            for j, policy in enumerate(self.policy_list):
                print(f"Trial #{tr:d} - policy {policy:s}", end="\r")
                # Simulate a trajectory of the SIS dynamical system under
                # various control strategies
                system = SISDynamicalSystem(
                    X_init, A, self.param_dict, self.cost_dict)
                data = system.simulate_policy(
                    policy, self.baselines_dict, self.sim_dict, plot=False)
                assert result[j]['name'] == policy
                result[j]['dat'].append(data)
        return result
