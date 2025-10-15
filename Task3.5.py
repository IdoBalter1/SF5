from Main import Network, assortative_network_sample_main,naive_create_network, gnp_two_stage, dfs, configuration_model, get_degree_distribution,simulate_epidemic,estimate_outbreak_disjointset_from_network, fast_disjoint_set
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from collections import defaultdict
from itertools import combinations
import math
import timeit
from scipy.cluster.hierarchy import DisjointSet
import networkx as nx
print("Task 3.5: Fast Disjoint Set2")
"""lambda_vals = np.linspace(0,0.3,50)
num_nodes = 10000
k = 20
x = random.randint(1, 10000)
for p in np.linspace(0,0.6,20):
  G = nx.watts_strogatz_graph(n=num_nodes, k=k, p=p)
  print("G")
  network = Network.from_networkx_to_custom(G)
  cluster_l = fast_disjoint_set(network,lambda_vals,x)

    
  plt.figure()
  plt.plot(lambda_vals, cluster_l, label='Fast Disjoint Set', marker='o', alpha=0.7)
  #plt.plot(lambda_vals, lambda_sizes, label='Estimate Outbreak Sizes', marker='x', linestyle='--')
  plt.show()"""
if __name__ == "__main__":
    network = Network.from_poisson(20, 10000)
    p_critical = 0.05
    lambda_values = np.linspace(0, 0.15, 100)

    num_simulation_runs = 25
    # Store the results from each run. Each element will be a list of 50 sizes.
    all_runs_results = []

    # It's better to use the same 'x' for all runs if you are averaging
    # for a specific node's perspective, or a new 'x' if you want to average
    # over different node perspectives AND different shuffles.
    # For consistency with the idea of averaging over shuffles for a given x:
    # x_node = random.randint(0, network.num_nodes - 1) if network.num_nodes > 0 else 0

    print(f"Running {num_simulation_runs} simulations...")
    for i in range(num_simulation_runs):
        # If you want to average over different starting nodes 'x' AND different shuffles:
        x_node = random.randint(0, network.num_nodes - 1) if network.num_nodes > 0 else 0
        print(f"  Run {i+1}/{num_simulation_runs} with x_node = {x_node}")
        # sizes_this_run will be a list of 50 expected cluster sizes, one for each lambda
        sizes_this_run = fast_disjoint_set(network, lambda_values, x_node)
        all_runs_results.append(sizes_this_run)

    # Convert the list of lists into a 2D NumPy array
    # Rows: simulation runs, Columns: lambda values
    all_runs_np = np.array(all_runs_results)

    # Calculate mean and std across the simulation runs (axis=0) for each lambda
    if all_runs_np.size > 0:
        mean_sizes_per_lambda = np.mean(all_runs_np, axis=0)
        std_devs_per_lambda = np.std(all_runs_np, axis=0)

        std_over_mean_sizes = []
        for i in range(len(lambda_values)):
            if mean_sizes_per_lambda[i] != 0:
                std_over_mean_sizes.append(std_devs_per_lambda[i] / mean_sizes_per_lambda[i])
            else:
                std_over_mean_sizes.append(0)
            # Optional: print per-lambda results
            # print(f"Lambda: {lambda_values[i]:.3f}, Mean Size: {mean_sizes_per_lambda[i]:.2f}, Std: {std_devs_per_lambda[i]:.2f}")
    else:
        print("Warning: No data generated from fast_disjoint_set runs.")
        mean_sizes_per_lambda = np.zeros(len(lambda_values))
        std_over_mean_sizes = np.zeros(len(lambda_values))


    plt.figure(figsize=(12, 5)) # Adjusted figure size

    plt.subplot(1, 2, 1)
    plt.plot(lambda_values, mean_sizes_per_lambda, 'bo-') # Use mean_sizes_per_lambda
    plt.xlabel('Lambda')
    plt.ylabel('Mean Size of Tracked Node Cluster')
    plt.title('Mean Size vs Lambda')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.axvline(x=p_critical, color='g', linestyle='--', label=f'Critical Lambda ({p_critical:.2f})')
    plt.plot(lambda_values, std_over_mean_sizes, 'ro-')
    plt.xlabel('Lambda')
    plt.ylabel('Std/Mean Size')
    plt.title('Std/Mean Size vs Lambda')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()



