from Main import Network, naive_create_network, gnp_two_stage, dfs, configuration_model, get_degree_distribution,simulate_epidemic
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from collections import defaultdict
from itertools import combinations
import math
import timeit

# Parameters
num_nodes = 10000
mean_degree = 20
num_trials = 100
num_weeks = 50
lambda_values = np.linspace(0.01, 0.3, 20)

# Store average final R counts
avg_final_R = []

# Generate fixed degree sequence (Poisson)
deg_seq = np.random.poisson(mean_degree, size=num_nodes)
net = Network.from_degree_sequence(deg_seq)

for lam in lambda_values:
    final_Rs = []
    for _ in range(num_trials):
        initial_infected = random.sample(range(num_nodes), k=10)
        S, I, R = simulate_epidemic(net, initial_infected, lam, num_weeks)
        final_Rs.append(R[-1])  # total recovered = total infected
    avg_final_R.append(np.mean(final_Rs))

# Normalize by population size (optional)
avg_final_R_frac = [r / num_nodes for r in avg_final_R]

# Plot
plt.figure(figsize=(8, 5))
plt.plot(lambda_values, avg_final_R_frac, marker='o')
plt.xlabel("Infection probability λ")
plt.ylabel("Final fraction infected (R/N)")
plt.title("Final outbreak size vs λ")
plt.grid(True)
plt.show()