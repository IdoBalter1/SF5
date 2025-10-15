from Main import Network, naive_create_network,assortative_network_sample_main,convert_to_network, gnp_two_stage, dfs, configuration_model, get_degree_distribution,simulate_epidemic,estimate_outbreak_disjointset_from_network
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from collections import defaultdict
from itertools import combinations
import math
import timeit
import networkx as nx

network = Network.from_poisson(20,10000)
network1 = Network.from_geometric(1/21,10000)
degrees = network.degrees()
num_nodes = network.num_nodes
sorted_degrees = sorted(range(len(degrees)), key=lambda i: degrees[i], reverse=True) 
sorted_degrees_reverse = sorted(range(len(degrees)), key=lambda i: degrees[i], reverse=False)
degrees_1 = network1.degrees()
num_nodes_1 = network1.num_nodes
# Generate assortative network
sorted_degrees1 = sorted(range(len(degrees_1)), key=lambda i: degrees_1[i], reverse=True)
sorted_degrees_reverse1 = sorted(range(len(degrees_1)), key=lambda i: degrees_1[i], reverse=False)

print(sorted_degrees_reverse1[0:10])
print(sorted_degrees_reverse[0:10])
for i in range(10):
    print(f"Node {i} in original network has degree {degrees[sorted_degrees1[i]]}")
    print(f"Node {i} in assortative network has degree {degrees_1[sorted_degrees_reverse1[i]]}")




# Corrected: Sample from the actual nodes in the network
"""initial_infected_nodes = random.sample(list(new_net.adj.keys()), 1)

S_counts, I_counts, R_Counts = simulate_epidemic(new_net, initial_infected_nodes, 0.2, 150)

# Ensure the outputs are numpy arrays for plotting
S_counts = np.array(S_counts)
I_counts = np.array(I_counts)
R_Counts = np.array(R_Counts)

plt.plot(S_counts, label='Susceptible')
plt.plot(I_counts, label='Infected')
plt.plot(R_Counts, label='Recovered')

# Add text to the plot
plt.text(0.02, 0.98, f'Assortativity: {r:.4f}\nClustering Coefficient: {c:.4f}', 
         transform=plt.gca().transAxes, verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.xlabel('Time')
plt.ylabel('Number of individuals')
plt.title('Epidemic Simulation on Assortative Network')
plt.legend()
plt.show()
"""