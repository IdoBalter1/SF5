from Main import Network, naive_create_network, gnp_two_stage, dfs, configuration_model, get_degree_distribution,simulate_epidemic, estimate_x_vector,compute_s_i,simulate_epidemic,estimate_outbreak_disjointset_from_network
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from collections import defaultdict
from itertools import combinations
import math
import timeit
from scipy.cluster.hierarchy import DisjointSet
def plot_suscetpible_infected_recovered_counts(I_counts,S_counts,R_counts, title):
    plt.figure(figsize=(10, 6))
    plt.plot(I_counts, marker='o', linestyle='-', color='b')
    plt.plot(R_counts, marker='x', linestyle='-', color='g')
    plt.plot(S_counts, marker='s', linestyle='-', color='r')
    plt.legend(['Infected', 'Recovered', 'Susceptible'])
    plt.title(title)
    plt.xlabel('Week')
    plt.ylabel('Number of Individuals')
    plt.grid()
    plt.show()


def vaccination_simulation(network, lam, percent_vaccinated):

    # Select nodes to vaccinate
    num_nodes = network.num_nodes
    vaccinated_nodes = np.random.choice(num_nodes, int(num_nodes * percent_vaccinated), replace=False)
    print(len(vaccinated_nodes), "nodes vaccinated")
    # Create a new network by removing vaccinated nodes
    reduced_net = Network(num_nodes)
    for i in range(num_nodes):
        if i in vaccinated_nodes:
            continue
        for j in network.neighbors(i):
            if j not in vaccinated_nodes:
                reduced_net.add_edge(i, j)

    # Pick initial infected from the remaining nodes
    susceptible_nodes = [i for i in range(num_nodes) if i not in vaccinated_nodes]
    initial_infected = random.sample(susceptible_nodes, k=20)

    # Run simulation
    sizes = estimate_outbreak_disjointset_from_network(reduced_net,lam)

    return sizes

"""
network = Network.from_poisson(20, 10000)
p_critical = 0.05
lambda_values = np.linspace(0,1,2)
mean_sizes = []

for lam in lambda_values:
    print(f"Running simulation for lambda = {lam:.2f}")
    all_sizes = []
    for _ in range(1):
        sizes = vaccination_simulation(network = Network.from_poisson(20,10000), lam = lam, percent_vaccinated = 0.4)
        all_sizes.extend(sizes)
    mean = np.mean(all_sizes)
    mean_sizes.append(mean)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(lambda_values, mean_sizes, 'bo-')
plt.xlabel('Lambda')
plt.ylabel('Mean Size')
plt.title('Mean Size vs Lambda')
plt.tight_layout()
plt.show()
"""


lambda_values = np.linspace(0, 0.6, )  # or however many steps you want
final_recovered_counts = []

def vaccination_simulation1(network, lam, percent_vaccinated):
    np.random.seed(42)

    num_nodes = network.num_nodes
    vaccinated_nodes = np.random.choice(num_nodes, int(num_nodes * percent_vaccinated), replace=False)

    reduced_net = Network(num_nodes)
    for i in range(num_nodes):
        if i in vaccinated_nodes:
            continue
        for j in network.neighbors(i):
            if j not in vaccinated_nodes and i < j:
                reduced_net.add_edge(i, j)

    susceptible_nodes = [i for i in range(num_nodes) if i not in vaccinated_nodes and reduced_net.degree(i) > 0]
    initial_infected = random.sample(susceptible_nodes, k=20)

    num_weeks = 50
    S_counts, I_counts, R_counts = simulate_epidemic(reduced_net, initial_infected, lam, num_weeks)

    final_R = R_counts[-1]  # <- this is what you want to plot
    return final_R

# Run for different lambda values
network = Network.from_poisson(20, 10000)
for lam in lambda_values:
    print(f"Running simulation for lambda = {lam:.2f}")
    final_R = vaccination_simulation1(network, lam=lam, percent_vaccinated=0.20)
    final_recovered_counts.append(final_R)

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(lambda_values, final_recovered_counts, 'bo-', label='Total Infected (R)')
plt.xlabel('Lambda (λ)')
plt.ylabel('Total Infected (R)')
plt.title('Total Infected vs λ (20% Vaccination)')
plt.grid(True)
plt.legend()
plt.show()
