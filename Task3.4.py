from Main import Network, naive_create_network,fast_disjoint_set,assortative_network_sample_main,convert_to_network, gnp_two_stage, dfs, configuration_model, get_degree_distribution,simulate_epidemic,estimate_outbreak_disjointset_from_network
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from collections import defaultdict
from itertools import combinations
import math
import timeit
import networkx as nx

print("Tswewewt.4: Assortatsdfsdfive Network Sample Maisdfsdfn with Averaging")
network = Network.from_poisson(20, 10000)

lambda_values = [0.02,0.03,0.05,0.1,0.2,0.4,0.7,1]
print("h")
meansizes = []
for lam in lambda_values:
    print(f"Processing lambda: {lam:.3f}")  # Show progress
    sizes = estimate_outbreak_disjointset_from_network(network, lam)
    sizes = np.array(sizes)  # Ensure sizes is a NumPy array
    mean_size = np.mean(sizes)
    meansizes.append(mean_size)

plt.figure(figsize=(10, 6))
plt.plot(lambda_values, meansizes, label='Mean Outbreak Size', color='blue')
plt.xlabel('Lambda Values')
plt.ylabel('Mean Outbreak Size')
plt.title('Mean Outbreak Size vs Lambda Values')
plt.legend()
plt.grid()
plt.show()


