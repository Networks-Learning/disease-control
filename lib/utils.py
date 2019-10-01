from collections import defaultdict
import numpy as np


def computed_infection_time_per_district(sir_object):
    """
    Return a dict keyed by district contained in node metadata of the graph `sir.G`. Each value
    is a list of all sorted infection times in the district.
    """
    district_inf_times = defaultdict(list)
    # Iterate over infection times of each node in the simulation
    for i, t in enumerate(sir_object.inf_occured_at):
        if (t > 0) and (t < np.inf):
            node_district = sir_object.G.node[sir_object.idx_to_node[i]]['district']
            district_inf_times[node_district].append(t)
    # Format as dict and sort values
    district_inf_times = dict(district_inf_times)
    for k, v in district_inf_times.items():
        district_inf_times[k] = sorted(v)
    return district_inf_times
