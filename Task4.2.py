from Main import Network, estimate_x_vector
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from collections import defaultdict
from itertools import combinations
import math
import timeit
from scipy.cluster.hierarchy import DisjointSet
average_degree = 20
n = 10000
p = 1 / (average_degree + 1)
lam_vals = np.linspace(0,0.3,10) # Slightly above critical value

# Create networks
netp = Network.from_poisson(average_degree, n)
netg = Network.from_geometric(p, n)
for lam in lam_vals:
    # Estimate x_i = 1 - s_i
    xp = estimate_x_vector(network=netp, lam=lam, max_iter=100, tolerance=1e-6)
    xg = estimate_x_vector(network=netg, lam=lam, max_iter=100, tolerance=1e-6)

    # Get degrees
    kp = netp.degrees()
    kg = netg.degrees()

    # Plot x_i vs degree for Poisson
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(kp, xp, alpha=0.4, s=10)
    plt.xlabel("Degree $k_i$")
    plt.ylabel("Infection probability $x_i$")
    plt.title("Poisson Network (λ = {:.2f})".format(lam))

    # Plot x_i vs degree for Geometric
    plt.subplot(1, 2, 2)
    plt.scatter(kg, xg, alpha=0.4, s=10, color='orange')
    plt.xlabel("Degree $k_i$")
    plt.ylabel("Infection probability $x_i$")
    plt.title("Geometric Network (λ = {:.2f})".format(lam))

    plt.tight_layout()
    plt.show()
