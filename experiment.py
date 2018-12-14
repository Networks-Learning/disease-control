
from dynamics import SISDynamicalSystem


'''
Class for easier handling of experiments.
Directly incorporates running the dynamical system simulation.

(Certain default values are set here to reduce clutter.)
'''

class Experiment:

    def __init__(self, name,
                       policy_dict=None,
                       sim_dict=None,
                       param_dict=None, 
                       cost_dict=None, 
                       baselines_dict=None):

        # defaults (Qx = 1, Qlam = 300)
        # baselines
        self.policy_dict = {
            'SOC': True,
            'TR': True,
            'TR-FL': True,
            'MN': True,
            'MN-FL': True,
            'LN': True,
            'LN-FL': True,
            'LRSR': True,
            'MCM': True,
        }

        # simulation settings
        self.sim_dict = {
            'total_time' : 10.00,
            'trials_per_setting' : 10
        }

        # model parameters
        self.param_dict = {
            'beta':  6.0,
            'gamma': 5.0,
            'delta': 1.0,
            'rho':   5.0,
            'eta':   1.0
        }

        # loss function parameterization
        self.cost_dict = {
            'Qlam': 1.0,
            'Qx': 300.0
        }

        # proportional scaling for fair comparison
        self.baselines_dict = {
            'TR': 0.003,
            'MN': 0.0007,
            'LN': 0.0008,
            'LRSR': 22.0,
            'MCM': 1.0,  # TODO
            'FL_info': dict(N=356.1000, max_u=19.5352),
        }

        self.name = name

        # change defaults to given parameters
        self.update(policy_dict=policy_dict,
                    sim_dict=sim_dict,
                    param_dict=param_dict,
                    cost_dict=cost_dict,
                    baselines_dict=baselines_dict)

    '''Updates dictionaries'''
    def update(self, policy_dict=None,
                     sim_dict=None,
                     param_dict=None,
                     cost_dict=None,
                     baselines_dict=None):

        if policy_dict:
            self.policy_dict = policy_dict

        if sim_dict:
            self.sim_dict = sim_dict
        
        if param_dict:
            self.param_dict = param_dict

        if cost_dict:
            self.cost_dict = cost_dict

        if baselines_dict:
            self.baselines_dict = baselines_dict

    '''Runs experiments as stated. Returns summary.'''

    def run(self, A, X_init):

        '''Simulate trajectory of dynamical system under various heuristics'''
        system = SISDynamicalSystem(X_init, A, self.param_dict, self.cost_dict)

        # find all policies set to be simulated
        available_policies = [k for k in self.policy_dict.keys() if self.policy_dict[k]]
        data = [dict(dat=[], name=None) for _ in range(len(available_policies))]

        # simulate number of times
        for tr in range(self.sim_dict['trials_per_setting']):
            print("Trial # " + str(tr))
            
            # ...for every requested policy 
            for j, policy in enumerate(available_policies):
                
                data[j]['name'] = policy
                data[j]['dat'].append(
                    system.simulate_policy(policy, 
                                           self.baselines_dict,
                                           self.sim_dict,
                                           plot=False))
        return data
