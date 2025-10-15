from Main import Network, naive_create_network, gnp_two_stage, dfs, configuration_model, get_degree_distribution,simulate_epidemic
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from collections import defaultdict
from itertools import combinations
import math
import timeit

num_nodes = 10000
net = Network.from_poisson(20, num_nodes) 
max_iter = 100
tolerance = 1e-6
def compute_s_i(net,lam,max_iter,tolerance):
    nodes =list(net.adj)
    s = [np.random.random() for node in nodes]

    for x in range(max_iter):
        s_new = {}
        max_change = 0
        for i in nodes:
            prod = 1.0
            for j in net.neighbors(i):
                prod *= (1 - lam + lam * s[j])
            s_new[i] = prod
            max_change = max(max_change, abs(s_new[i] - s[i]))
        s = s_new
        if max_change < tolerance:
            break

    # Compute final expected number infected
    total_infected = sum(1 - si for si in s.values())
    return s, total_infected


num_nodes = 10000
net = Network.from_poisson(20, num_nodes) 
max_iter = 100
tolerance = 1e-6
mean_degree = 20
num_trials = 100
num_weeks = 50
lambda_values = np.linspace(0.01, 0.3, 20)
infected_analytic = []

for lam in lambda_values:
    _, total = compute_s_i(net,lam,max_iter,tolerance)
    infected_analytic.append(total / num_nodes)

avg_final_R = []

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

plt.plot(lambda_values, infected_analytic, label="Analytic Prediction")
plt.plot(lambda_values, avg_final_R_frac, label="Simulation", linestyle='--')
plt.xlabel("λ (infection probability)")
plt.ylabel("Final infected fraction")
plt.title("SIR: Simulation vs Analytic Fixed Point")
plt.legend()
plt.grid(True)
plt.show()