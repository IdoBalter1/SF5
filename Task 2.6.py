from Main import Network, naive_create_network, gnp_two_stage, dfs, configuration_model, get_degree_distribution
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, geom
import random
import time
# Generate a poisson graph with mean between 0 and 2
#create a list of average degrees between 0 and 2
num_nodes = 2**10
average_degrees = np.linspace(0, 2,50)
average_degree_list_poisson = []
average_degree_list_geometric = []
for average_degree in average_degrees:
    print(average_degree)
    lam = average_degree
    p = 1/(average_degree + 1)
    run_means_poisson = []
    run_means_geometric = []
    for _ in range(20):
        x = np.random.randint(0, num_nodes)
        net1 = Network.from_poisson(lam, num_nodes)
        net2 = Network.from_geometric(p, num_nodes)
        visited_counts_poisson = []
        visited_counts_geometric = []
        for _ in range(1000):
            visited_poisson = dfs(net1.adj, x)
            visited_counts_poisson.append(len(visited_poisson))
            visited_geometric = dfs(net2.adj, x)
            visited_counts_geometric.append(len(visited_geometric))
        run_means_poisson.append(np.mean(visited_counts_poisson))
        run_means_geometric.append(np.mean(visited_counts_geometric))
    average_degree_list_poisson.append(np.mean(run_means_poisson))
    average_degree_list_geometric.append(np.mean(run_means_geometric))

# Plot the results
plt.figure(figsize=(12, 6))     
plt.plot(average_degrees, average_degree_list_poisson, marker='o', label='Poisson Degree Distribution')
plt.plot(average_degrees, average_degree_list_geometric, marker='o', label='Geometric Degree Distribution')
#plot a vertical line at x = 2/3
plt.axvline(x=1/2, color='r', linestyle='--', label='CRITICAL GEOMETRIC THRESHOLD (2/3)')
plt.axvline(x=1, color='g', linestyle='--', label='CRITICAL POISSON THRESHOLD (1)')
plt.xlabel('Average Degree')
plt.ylabel('Average Size of Component Containing Node')
plt.title('Component Size vs Average Degree for Poisson and Geometric Distributions')
plt.legend()
plt.grid(True)
plt.show()


