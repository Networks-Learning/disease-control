from dynamics import SISDynamicalSystem


class Experiment:
    """
    Class for easier handling of experiments.
    Directly incorporates running the dynamical system simulation.

    Attributes:
    -----------
    policy_list : list
        List of policies to run for the experiment. `SOC` should be first to set front-loading
        parameters of baselines.
    sim_dict : dict
        Simulation parameters. Should have keys:
        - 'total_time': the time to run each simulation
        - 'trials_per_setting': the number of trials for each simulation
    param_dict : dict
        Model parameters. Should have keys:
        - 'beta': infection rate
        - 'gamma': reduction in infection rate
        - 'delta': spontaneous recovery rate
        - 'rho': treatement recovery rate
        - 'eta': exponential decay for SOC strategiy
    cost_dict : dict
        Costs of SOC strategiy. Should have keys:
        - 'Qlam': cost of treatement
        - 'Qx': cost of infection
    baselines_dict : dict
        Parameters for baseline strategies. Should have keys:
        - each policy with its corresponding scaling parameter
        - 'FL_info': a dict with keys:
            - 'N': Maximum number of treatement
            - 'max_u': Maximum intensity
    """

    def __init__(self, name, policy_list, sim_dict, param_dict, cost_dict, baselines_dict):
        # Default policies to run
        self.policy_list = ['SOC']
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
            'LRSR': 22.807,
            'MCM': 22.807,
            'FL_info': {'N': None, 'max_u': None},
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
            assert policy_list[0] == 'SOC', "Strategy `SOC` must be run first to set FL info."
            self.policy_list = policy_list
        if sim_dict:
            self.sim_dict = sim_dict
        if param_dict:
            self.param_dict = param_dict
        if cost_dict:
            self.cost_dict = cost_dict
        if baselines_dict:
            self.baselines_dict = baselines_dict

    def update_fl_info(self, data):
        """
        Update the `FL_info` dict based on the current data. `data` must the collected data from
        a run of a SISDynamicalSystem.
        """
        assert data['info']['policy'] == 'SOC'
        # Extract the maximum value of the control intensity
        max_u = max([max([proc.value_at(t) for t in proc.arrival_times]) for proc in data['u']])
        # Extract the number of treatements
        n_treatement = sum([proc.get_current_value() for proc in data['Nc']])
        # Update the FL info dict
        if self.baselines_dict['FL_info']['N'] is None:
            self.baselines_dict['FL_info']['max_u'] = max_u
            self.baselines_dict['FL_info']['N'] = n_treatement
        elif n_treatement > self.baselines_dict['FL_info']['N']:
            self.baselines_dict['FL_info']['max_u'] = max_u
            self.baselines_dict['FL_info']['N'] = n_treatement

    def run(self, A, X_init):
        """
        Run the experiment and return a summary as a list of dict.
        """
        n_trials = self.sim_dict['trials_per_setting']
        # Initialize the result object
        result = [{"dat": [], "name": policy} for policy in self.policy_list]
        # Simulate every requested policy
        for j, policy in enumerate(self.policy_list):
            print(f"=== Policy: {policy:s}...")
            # ...for many trials
            for tr in range(n_trials):
                print(f"  - Trial {tr+1:d}/{n_trials}")
                # Simulate a trajectory of the SIS dynamical system under
                # various control strategies
                system = SISDynamicalSystem(
                    X_init, A, self.param_dict, self.cost_dict)
                data = system.simulate_policy(
                    policy, self.baselines_dict, self.sim_dict, plot=False)
                # Add the policy name to the info dict of the run
                data['info']['policy'] = policy
                # Add the FL_info parameters to the info dict of the run
                data['info']['FL_info'] = self.baselines_dict['FL_info'].copy()
                # Make sure the policy is the correct one and add the data to result
                assert result[j]['name'] == policy
                result[j]['dat'].append(data)
                # If policy is SOC, we update the FL info dict
                if policy == 'SOC':
                    self.update_fl_info(data)
        return result
