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


    # Fit a 3rd-degree polynomial for Poisson
    coeffs_p = np.polyfit(kp, xp, deg=3)
    poly_p = np.poly1d(coeffs_p)
    x_line_p = np.linspace(min(kp), max(kp), 300)
    y_line_p = poly_p(x_line_p)

    # Fit a 3rd-degree polynomial for Geometric
    coeffs_g = np.polyfit(kg, xg, deg=3)
    poly_g = np.poly1d(coeffs_g)
    x_line_g = np.linspace(min(kg), max(kg), 300)
    y_line_g = poly_g(x_line_g)

    # Plot both subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Poisson plot
    axs[0].scatter(kp, xp, alpha=0.2, s=5, label='Data')
    axs[0].plot(x_line_p, y_line_p, color='red', label='Best Fit')
    axs[0].set_title(f'Poisson Network ($\lambda$ = {lam:.2f})')
    axs[0].set_xlabel('Degree $k_i$')
    axs[0].set_ylabel('Infection probability $x_i$')
    axs[0].legend()

    # Geometric plot
    axs[1].scatter(kg, xg, alpha=0.2, color='orange', s=5, label='Data')
    axs[1].plot(x_line_g, y_line_g, color='red', label='Best Fit')
    axs[1].set_title(f'Geometric Network ($\lambda$ = {lam:.2f})')
    axs[1].set_xlabel('Degree $k_i$')
    axs[1].set_ylabel('Infection probability $x_i$')
    axs[1].legend()

    plt.tight_layout()
    plt.show()
