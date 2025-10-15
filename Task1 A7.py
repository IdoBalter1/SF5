from Main import Network, naive_create_network, gnp_two_stage,dfs
import numpy as np
import random
from itertools import combinations
from collections import defaultdict
from matplotlib import pyplot as plt
import math
import timeit
n = 4096
p_crit = 1/(n-1)
avg_sizes = []
P_values = np.linspace(0, 0.001, 30) 
for p in P_values:  
    print(f"p: {p}")
    visited_counts = []
    for i in range(1000):
        net, edge_count, adjacency_matrix = gnp_two_stage(n, p)
        visited = dfs(net.adj,1)
        visited_counts.append(len(visited))
    avg_sizes.append(np.mean(visited_counts))  # Average number of edges for this p


plt.figure(figsize=(10,6))
plt.plot(P_values, avg_sizes, marker='o', label="Avg. component size") 
plt.axvline(1/(n-1), color='red', linestyle='--', label='p ≈ 1/(n−1)')
plt.xlabel("Edge probability (p)")
plt.ylabel("Avg. size of component containing node 1")
plt.title("Component growth in G(n, p)")
plt.legend()
plt.grid(True)
plt.show()