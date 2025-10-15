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

from scipy.stats import poisson

fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# Poisson configuration model
axs[0].hist(degrees_p, bins=range(min(degrees_p), max(degrees_p) + 1), density=True,
            alpha=0.6, color='skyblue', edgecolor='black', label='Empirical')
k_vals_p = np.arange(min(degrees_p), max(degrees_p) + 1)
theory_pmf_p = poisson.pmf(k_vals_p, mu=10)
axs[0].plot(k_vals_p, theory_pmf_p, 'r-', marker='o', label='Poisson PMF (λ=10)')
axs[0].set_title("Degree Distribution: Poisson(λ=10)")
axs[0].set_xlabel("Degree")
axs[0].set_ylabel("Probability")
axs[0].legend()
axs[0].grid(True)

# Geometric configuration model
p = 1/11
S_g = configuration_model(k = np.random.geometric(p, num_nodes) - 1, num_nodes = num_nodes)
degrees_g = get_degree_distribution(S_g, num_nodes)
mean_degrees_g = np.mean(degrees_g)
print(f"Mean degree for Geometric(p=1/11): {mean_degrees_g:.2f}")
k_vals_g = np.arange(min(degrees_g), max(degrees_g) + 1)
theory_pmf_g = p * (1 - p) ** k_vals_g
axs[1].hist(degrees_g, bins=range(min(degrees_g), max(degrees_g) + 1), density=True,
            alpha=0.6, color='skyblue', edgecolor='black', label='Empirical')
axs[1].plot(k_vals_g, theory_pmf_g, 'r-', marker='o', label='Geometric PMF (shifted, p=1/11)')
axs[1].set_title("Degree Distribution: Shifted Geometric(p=1/11)")
axs[1].set_xlabel("Degree")
axs[1].set_ylabel("Probability")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()
# Compare this snippet from Task2.1.py:
# from Main import Network, naive_create_network, sample_create__network, gnp_two_stage,dfs,configuration_model, get_degree_distribution
# import numpy as np
