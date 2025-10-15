from Main import Network, naive_create_network, gnp_two_stage, dfs, configuration_model, get_degree_distribution,simulate_epidemic, estimate_x_vector,compute_s_i,simulate_epidemic,estimate_outbreak_disjointset_from_network,vaccinate_disjoint_set
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from collections import defaultdict
from itertools import combinations
import math
import timeit
from scipy.cluster.hierarchy import DisjointSet
net = Network.from_poisson(20, 10000)
lambda_values = np.linspace(0, 1, 10)
mean_sizes = []

for lam in lambda_values:
    all_sizes = []
    for _ in range(10):
        sizes = vaccinate_disjoint_set(network = net, lam = lam,percent = 0.4)
        all_sizes.extend(sizes)
    mean = np.mean(all_sizes)
    print(f"Lambda: {lam}, Mean Size: {mean}")
    mean_sizes.append(mean)

plt.figure(figsize=(10, 4))
plt.plot(1, 2, 1)
plt.plot(lambda_values, mean_sizes, 'bo-')
plt.xlabel('Lambda')
plt.ylabel('Mean Size')
plt.title('Mean Size vs Lambda')
plt.show()
