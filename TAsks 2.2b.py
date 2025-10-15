from Main import Network, naive_create_network, gnp_two_stage,dfs,configuration_model, get_degree_distribution,assortative_negative_network, assortative_network
import numpy as np
import matplotlib.pyplot as plt
import random
from itertools import combinations
from collections import defaultdict
from matplotlib import pyplot as plt
import math
import timeit

"""
Generate a random number between 0 and 9999. This is the node that we will evaluate.
initilaise an empty list for the amount of neighbours.
Initialise another emptty list for the amount of neighbours of the random neighbour.
Find the neighboours of this node.
add the amount of neighbours to the list.
choose a ranodm neighbour from the list of neighbours.
find the amount of neighbours of this neighbour.
add the amkount of neighbours to the list of the random neighbour.
repeat and average
"""
#Result: Average degree of random node is 10.98, average degree of friend is 20.90 for 1000000 trials.
import random
import numpy as np
import matplotlib.pyplot as plt


num_nodes = 10000
p = 1/11
geometric_network = Network.from_geometric(p , num_nodes=num_nodes)
print("Expected mean degree:", (1 - p) / p)



import numpy as np
import matplotlib.pyplot as plt
import random

num_nodes = 10000
samples = 10

p = 1 / 11

net = Network.from_power_law(alpha = 1.8, num_nodes=num_nodes)
new_net, degrees = assortative_network(net)  # Biased (e.g. negatively assortative) network

# Arrays to store degree data
geom_self_k_array = []
geom_friend_k_array = []
geom_self_k_biased = []
geom_friend_k_biased = []

# Pre-sample nodes
random_self = np.random.randint(0, num_nodes, size=samples)

for i in random_self:
    # --- Unbiased network ---
    g_i = i
    while not len(net.neighbors(g_i)):
        geom_self_k_array.append(0)  # Include zero for isolated node
        g_i = np.random.randint(0, num_nodes)

    neighbors = list(net.neighbors(g_i))
    geom_self_k_array.append(len(neighbors))
    friend = np.random.choice(neighbors)
    geom_friend_k_array.append(len(net.neighbors(friend)))

    # --- Biased network ---
    g_j = i
    while not len(new_net.neighbors(g_j)):
        geom_self_k_biased.append(0)
        g_j = np.random.randint(0, num_nodes)

    neighbors_biased = list(new_net.neighbors(g_j))
    geom_self_k_biased.append(len(neighbors_biased))
    friend_biased = np.random.choice(neighbors_biased)
    geom_friend_k_biased.append(len(new_net.neighbors(friend_biased)))

# --- Results ---
print(f"Average degree of random nodes: {np.mean(geom_self_k_array):.2f}")
print(f"Average degree of their friends: {np.mean(geom_friend_k_array):.2f}")
print(f"Average degree of random nodes (biased): {np.mean(geom_self_k_biased):.2f}")
print(f"Average degree of their friends (biased): {np.mean(geom_friend_k_biased):.2f}")

# --- Plotting ---
plt.figure(figsize=(14, 6))

# Unbiased
plt.subplot(1, 2, 1)
plt.hist(geom_self_k_array, bins=30, alpha=0.6, label='Random Nodes')
plt.hist(geom_friend_k_array, bins=30, alpha=0.6, label='Friends')
plt.axvline(np.mean(geom_self_k_array), color='blue', linestyle='--', label=f'Avg Random: {np.mean(geom_self_k_array):.2f}')
plt.axvline(np.mean(geom_friend_k_array), color='orange', linestyle='--', label=f'Avg Friend: {np.mean(geom_friend_k_array):.2f}')
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.title('Unbiased Geometric Network')
plt.legend()

# Biased
plt.subplot(1, 2, 2)
plt.hist(geom_self_k_biased, bins=30, alpha=0.6, label='Random Nodes (Biased)')
plt.hist(geom_friend_k_biased, bins=30, alpha=0.6, label='Friends (Biased)')
plt.axvline(np.mean(geom_self_k_biased), color='blue', linestyle='--', label=f'Avg Random: {np.mean(geom_self_k_biased):.2f}')
plt.axvline(np.mean(geom_friend_k_biased), color='orange', linestyle='--', label=f'Avg Friend: {np.mean(geom_friend_k_biased):.2f}')
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.title('Biased Geometric Network')
plt.legend()

plt.tight_layout()
plt.show()


