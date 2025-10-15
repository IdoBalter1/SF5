import numpy as np
import random
from itertools import combinations
from collections import defaultdict
from matplotlib import pyplot as plt
import math
from scipy.cluster.hierarchy import DisjointSet
from scipy.stats import binom
import networkx as nx
from Main import Network, naive_create_network, gnp_two_stage, dfs, configuration_model, get_degree_distribution, assortative_negative_network,assortative_network,assortative_network_sample,assortative_network_sample_main,convert_to_network,fast_disjoint_set
print("saved")
# These global variables are used by the function
std_values_low_conf = np.linspace(4.5,20,40)
std_values_low_mid_conf = np.linspace(21,60,40)
std_mid_conf = np.linspace(65,100,40)
std_high_conf = np.linspace(105,300,40)
std_high_high_conf = np.linspace(350,8000,50)
std_values  = np.concatenate((std_values_low_conf,std_values_low_mid_conf, std_mid_conf,std_high_conf,std_high_high_conf))
trials = 100
network_p = Network.from_poisson(20, 10000)
network_g = Network.from_geometric(1/21, 10000) # Fixed typo: netowrk_g -> network_g
lambda_values = np.linsapce()
sizes_array = []
for std in std_values:
    print(f"Processing std_dev_varaince_subcritical: {std:.2f}") # Optional: for progress tracking
    means = []
    for i in range(trials):
        new_net = assortative_network_sample_main(network_p, std, 1000, 'positive')
        x = random.randint(0,9999)
        sizes = fast_disjoint_set(new_net) # fast disjoin set returns a list of sizes cluster for each value of lambda
        sizes_array.append(sizes)



