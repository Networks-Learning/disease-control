from collections import defaultdict, Counter
import numpy as np


def compute_infection_time_per_district(inf_arr, district_arr):
    """
    Return a dict keyed by district contained in node metadata of the graph `sir.G`. Each value
    is a list of all sorted infection times in the district.
    """
    district_inf_times = defaultdict(list)
    # Iterate over infection times of each node in the simulation
    for i, (inf_time, district) in enumerate(zip(inf_arr, district_arr)):
        if (inf_time > 0) and (inf_time < np.inf):
            district_inf_times[district].append(inf_time)
    # Format as dict and sort values
    district_inf_times = dict(district_inf_times)
    for k, v in district_inf_times.items():
        district_inf_times[k] = np.array(sorted(v))
    return district_inf_times


def compute_r0_per_country(inf_time_arr, infector_arr, country_arr):
    """
    Compute the basic reproduction number R0 per country for the given data.

    inf_time_arr : array-like (shape: (N,))
        Infection time of each of the N nodes
    infector_arr : array-like (shape: (N,))
        Index of the
    """
    # Count of secondary infections: {node_idx: num of infected neighbors}
    infector_count = Counter(infector_arr)
    # Indices of infected nodes
    infected_node_indices = np.where(np.array(inf_time_arr) < np.inf)[0]
    # Initialize the list of number of secondary infections per country
    country_count = {country: list() for country in set(country_arr)}
    # For each infected node, add its number of secondary case to its country
    for u_idx in infected_node_indices:
        u_country = country_arr[u_idx]
        inf_count = infector_count[u_idx]
        country_count[u_country].append(inf_count)
    country_count = dict(country_count)
    # Compute R0 as the mean number of secondary case for each country
    countru_r0_dict = {country: np.mean(count) if len(count) > 0 else 0.0 for country, count in country_count.items()}
    return countru_r0_dict
