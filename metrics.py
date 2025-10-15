import random
import numpy as np
from epidemic import estimate_outbreak_disjointset_from_network


def vaccinate_disjoint_set(network, lam, percent):
    print(network.edge_count, "edges before vaccination")
    num_nodes = network.num_nodes
    print(num_nodes, "nodes in the network")
    num_vaccinated = int(num_nodes * percent)
    vaccinated_nodes = set(random.sample(range(num_nodes), num_vaccinated))
    new_net = type(network)(num_nodes)
    for node in range(num_nodes):
        if node in vaccinated_nodes:
            continue
        for neighbor in network.neighbors(node):
            if neighbor not in vaccinated_nodes:
                new_net.add_edge(node, neighbor)
    print(new_net.edge_count, "edges left after vaccination")
    sizes = estimate_outbreak_disjointset_from_network(new_net, lam)
    return sizes


def compute_s_i(net, lam, max_iter, tolerance):
    nodes = list(net.adj)
    s = [np.random.random() for _ in nodes]
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
    total_infected = sum(1 - si for si in s.values())
    return s, total_infected


def estimate_x_vector(network, lam, max_iter=100, tolerance=1e-5):
    s, _ = compute_s_i(network, lam, max_iter, tolerance)
    x = [1 - s[i] for i in range(network.num_nodes)]
    return x
