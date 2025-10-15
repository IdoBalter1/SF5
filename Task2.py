from Main import Network, naive_create_network, gnp_two_stage,dfs,configuration_model, get_degree_distribution
import numpy as np
import matplotlib.pyplot as plt
import random
from itertools import combinations
from collections import defaultdict
from matplotlib import pyplot as plt
import math
import timeit
lam = 10
num_nodes = 10000
S_p = configuration_model(k = np.random.poisson(lam, num_nodes), num_nodes = num_nodes)

degrees_p = get_degree_distribution(S_p, num_nodes)

# Step 3: Plot the histogram
plt.figure(figsize=(8, 5))
plt.hist(degrees_p, bins=range(min(degrees_p), max(degrees_p) + 1), density=True, alpha=0.7, color='skyblue', edgecolor='black')
plt.title("Degree Distribution (Poisson Configuration Model)")
plt.xlabel("Degree")
plt.ylabel("Probability")
plt.grid(True)
plt.show()

from scipy.stats import poisson

# Compute theoretical distribution
k_vals = np.arange(min(degrees_p), max(degrees_p) + 1)
theory_pmf = poisson.pmf(k_vals, mu=10)

# Plot overlay
plt.figure(figsize=(8, 5))
plt.hist(degrees_p, bins=range(min(degrees_p), max(degrees_p) + 1), density=True,
         alpha=0.6, color='skyblue', edgecolor='black', label='Empirical')
plt.plot(k_vals, theory_pmf, 'r-', marker='o', label='Poisson PMF (λ=10)')
plt.title("Degree Distribution: Empirical vs Poisson(λ=10)")
plt.xlabel("Degree")
plt.ylabel("Probability")
plt.legend()
plt.grid(True)
plt.show()