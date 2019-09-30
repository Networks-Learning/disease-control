import pandas as pd
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
import scipy.stats
from tqdm import tqdm
import scipy.optimize
import joblib
import networkx as nx

from dynamics import SISDynamicalSystem
from analysis import Evaluation
from stochastic_processes import StochasticProcess, CountingProcess

from helpers import HelperFunc


'''Construct network from input'''
df = pd.read_csv("data/contiguous-usa.txt", sep=" ", header=None)
df = df - 1
df = df.drop(columns=[2])
df_0, df_1 = df[0].values, df[1].values
N, M = max(np.max(df_0), np.max(df_1)) + 1, len(df_0)

# load X, u
print('load X, u...')
hf = HelperFunc()
filename = 'results_comparison_OPT_T_MN_complete_8_50__Q_1_400_.pkl'
data = joblib.load('temp_pickles/' + filename)
print('done.')



times = np.arange(1.0, 4.0, 0.1).tolist()
times2 = np.arange(1.001, 4.001, 0.1).tolist()
times = times + times + times2
times.sort()

trial_no = 1

for t in times:

    u_t = hf.sps_values(data[0][trial_no]['u'], t, summed=False)
    X_t = hf.sps_values(data[0][trial_no]['X'], t, summed=False)

    G = nx.from_pandas_edgelist(df.astype('i'), 0, 1, create_using=nx.Graph())

    # And a data frame with characteristics for your nodes
    # X = [0 if i % 4 == 0 else 1 for i in range(N)]
    # u = [i for i in range(N)]
    X = list(X_t)
    u = list(u_t)

    # Infection tags
    carac = pd.DataFrame({'ID': G.nodes(), 'myvalue': X})
    carac['myvalue'] = pd.Categorical(carac['myvalue'])

    pos = nx.spring_layout(G, k=0.04)


    nodes_to_u = {i: u[i] for i in G.nodes()}
    nodes_to_X = {i: X[i] for i in G.nodes()}

    X_to_edgecolors = ['red' if X[i] == 1.0 else 'black' for i in G.nodes()]
    X_to_linewidths = [2.0 if X[i] == 1.0 else 0.7 for i in G.nodes()]

    healthy_nodes = [i for i in G.nodes() if X[i] != 1.0]
    infected_nodes = [i for i in G.nodes() if X[i] == 1.0 and abs(u[i]) == 0.0]
    infected_treated_nodes = [i for i in G.nodes() if X[i] == 1.0 and abs(u[i]) != 0.0]
    # add one infected node with u = 0 to 'under treatment' for proper color mapping
    if infected_nodes:
        v = infected_nodes.pop()
        infected_treated_nodes.append(v)
    infected_treated_nodes_to_u = {i: u[i] for i in infected_treated_nodes}


    plt.figure(figsize=(6, 4))
    nx.draw_networkx_edges(G, pos, nodelist=list(nodes_to_u.keys()), alpha=0.4)
    nx.draw_networkx_nodes(G, pos, nodelist=infected_treated_nodes,
                        node_size=100,
                        node_color=list(infected_treated_nodes_to_u.values()),
                        cmap=plt.cm.Blues,
                        # node_color='blue',
                        linewidths=2.5,
                        edgecolors='black',
                        label='infected and targeted for treatment')
    nx.draw_networkx_nodes(G, pos, nodelist=infected_nodes,
                           node_size=100,
                           node_color='white',
                           # cmap=plt.cm.Blues,
                           linewidths=2.5,
                           edgecolors='black',
                           label='infected')
    nx.draw_networkx_nodes(G, pos, nodelist=healthy_nodes,
                        node_size=100,
                        node_color='white',
                        #    cmap=plt.cm.Blues,
                        linewidths=0.7,
                        edgecolors='black',
                        label='healthy')
    plt.axis('off')
    plt.legend(numpoints=1)
    plt.savefig('graphs/network_visualization_3_{}.png'.format(int(t * 1000)), frameon=False, format='png', dpi=600)
    # plt.show()
    plt.close('all')

