from Main import Network, naive_create_network, gnp_two_stage, dfs, configuration_model, get_degree_distribution,simulate_epidemic, estimate_x_vector,compute_s_i
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from collections import defaultdict
from itertools import combinations
import math
import timeit
from scipy.cluster.hierarchy import DisjointSet

def verify_disease_paradox(network, x_vector):
    """
    For each node i, compute its own infection probability x_i,
    and the average infection probability of its neighbors (kappa_i).
    Returns the average difference (kappa_i - x_i) over all nodes with degree > 0.
    """
    total_difference = 0
    count = 0

    for i in range(network.num_nodes):
        neighbors = network.neighbors(i)
        deg = len(neighbors)
        if deg == 0:
            continue  # skip degree 0 nodes
        x_i = x_vector[i]
        kappa_i = np.mean([x_vector[j] for j in neighbors])
        total_difference += (kappa_i - x_i)
        count += 1

    average_difference = total_difference / count
    return average_difference

lam = 0.1  # Choose λ > λ_c
network = Network.from_poisson(20, 10000)  # Create a network with Poisson degree distribution
s_vector, _ = compute_s_i(network, lam, max_iter=100, tolerance=1e-6)
x_vector = [1 - s for s in s_vector.values()]  # convert s_i to x_i

# Run disease paradox check
avg_diff = verify_disease_paradox(network, x_vector)
print(f"Average (friends' x - own x): {avg_diff:.4f}")


def compute_disease_paradox_deltas(network, x_vector):
    deltas = []
    for i in range(network.num_nodes):
        neighbors = network.neighbors(i)
        k_i = len(neighbors)
        if k_i == 0:
            continue  # skip isolated nodes
        x_i = x_vector[i]
        kappa_i = np.mean([x_vector[j] for j in neighbors])
        delta_i = kappa_i - x_i
        deltas.append(delta_i)
    return deltas
# Assume you've already run:
# s_vector, _ = compute_s_i(network, lam, max_iter=100, tolerance=1e-6)
# x_vector = [1 - s for s in s_vector.values()]

deltas = compute_disease_paradox_deltas(network, x_vector)

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(deltas, bins=50, color='skyblue', edgecolor='black')
plt.axvline(np.mean(deltas), color='red', linestyle='--', label=f"Mean = {np.mean(deltas):.4f}")
plt.title("Histogram of Δᵢ = Xᵢ − xᵢ (Disease Paradox)")
plt.xlabel("Δᵢ")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.show()

# Print mean delta
print(f"Mean Δᵢ: {np.mean(deltas):.4f}")
