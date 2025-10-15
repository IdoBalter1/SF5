from Main import Network, naive_create_network, gnp_two_stage
import numpy as np
import random
from itertools import combinations
from collections import defaultdict
from matplotlib import pyplot as plt

num_nodes = 100
for p in np.arange(0.1, 1.1, 0.1):    
    edge_counts = []  # Step size of 0.1
    mean = 0
    for i in range(1000):
        edge_count,adjecency_matrix = gnp_two_stage(num_nodes, p)
        edge_counts.append(edge_count)

    mean = np.mean(edge_counts)
    variance = np.var(edge_counts)
    theoretical_mean = num_nodes * (num_nodes - 1) / 2 * p
    theoretical_variance = num_nodes * (num_nodes - 1) / 2 * p * (1 - p)
    print(f"Probability: {p:.1f}, Mean: {mean:.2f}, Variance: {variance:.2f}")
    print(f"Theoretical Mean: {theoretical_mean:.2f}, Theoretical Variance: {theoretical_variance:.2f}")


    """plt.hist(edge_counts, bins=50, edgecolor='black')  # Adjust 'bins' as needed
    plt.title(f'Histogram of Number of Edges (n={num_nodes}, p={p:.1f})')
    plt.xlabel('Number of Edges')
    plt.ylabel('Frequency')
    plt.show()"""




