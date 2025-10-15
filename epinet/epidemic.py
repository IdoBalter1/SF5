import random
from scipy.cluster.hierarchy import DisjointSet
from scipy.stats import binom
import numpy as np


def dfs(graph, start):
    visited = set()
    stack = [start]
    result = []
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            result.append(vertex)
            stack.extend(graph[vertex] - visited)
    return result


def simulate_epidemic(network, initial_infected, lam, num_weeks):
    susceptible = set(network.adj)
    infected = set(initial_infected)
    recovered = set()
    susceptible -= infected

    S_counts = [len(susceptible)]
    I_counts = [len(infected)]
    R_counts = [len(recovered)]

    for week in range(num_weeks):
        new_infected = set()
        for node in infected:
            for neighbor in network.neighbors(node):
                if neighbor in susceptible and random.random() < lam:
                    new_infected.add(neighbor)
        susceptible -= new_infected
        recovered.update(infected)
        infected = new_infected

        S_counts.append(len(susceptible))
        I_counts.append(len(infected))
        R_counts.append(len(recovered))

        if len(infected) == 0:
            break

    return S_counts, I_counts, R_counts


def estimate_outbreak_disjointset_from_network(network, lam):
    edge_list = network.edge_list()
    C = DisjointSet(range(network.num_nodes))
    for i, j in edge_list:
        if np.random.rand() < lam:
            C.merge(i, j)
    sizes = [C.subset_size(i) for i in range(network.num_nodes)]
    return sizes


def fast_disjoint_set(network, lambda_values, x):
    cluster_size_k = []
    cluster_sizes_lambda = []
    edge_list = network.edge_list()
    np.random.shuffle(edge_list)
    C = DisjointSet(range(network.num_nodes))
    cluster_size_k.append(C.subset_size(x))
    for (i, j) in edge_list:
        C.merge(i, j)
        cluster_size_k.append(C.subset_size(x))

    for lam in lambda_values:
        k_values = np.arange(len(edge_list) + 1)
        pmf_values = binom.pmf(k_values, len(edge_list), lam)
        cluster_size_k_arr = np.array(cluster_size_k)
        cluster_size_lambda = np.sum(pmf_values * cluster_size_k_arr)
        cluster_sizes_lambda.append(cluster_size_lambda)

    return cluster_sizes_lambda


def peak_infection_metrics(network, initial_infected, lam, num_weeks):
    _, I_counts, _ = simulate_epidemic(network, initial_infected, lam, num_weeks)
    peak_size = max(I_counts) if I_counts else 0
    time_to_peak = I_counts.index(peak_size) if peak_size in I_counts else 0
    return peak_size, time_to_peak


def length_of_infection_period(network, initial_infected, lam, num_weeks):
    S_counts, I_counts, R_counts = simulate_epidemic(network, initial_infected, lam, num_weeks)
    for week in range(1, len(I_counts)):
        if I_counts[week] == 0:
            return week
    return num_weeks
