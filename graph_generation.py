import pandas as pd
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
import scipy.stats
from tqdm import tqdm
import scipy.optimize
import joblib

import networkx

G = networkx.random_regular_graph(5, 50, seed=0).to_undirected()
A = networkx.to_numpy_array(G)
