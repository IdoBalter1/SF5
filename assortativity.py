import numpy as np
import math
import random
from network import Network


def assortative_network(network):
    degrees = network.degrees()
    num_nodes = network.num_nodes
    sorted_degrees = sorted(range(len(degrees)), key=lambda i: degrees[i], reverse=True)
    new_net = Network(num_nodes)
    for idx_i, i in enumerate(sorted_degrees):
        degree = degrees[i]
        num_connections = 0
        if degree == 0:
            continue
        for idx_j in range(idx_i + 1, len(sorted_degrees)):
            j = sorted_degrees[idx_j]
            if degrees[j] == 0:
                continue
            new_net.add_edge(i, j)
            degrees[i] -= 1
            degrees[j] -= 1
            num_connections += 1
            if num_connections == degree or degrees[i] == 0:
                break
    return new_net, degrees


def assortative_network_sample(network, std_dev, max_attempts, value):
    degrees = network.degrees()
    num_nodes = network.num_nodes
    sorted_degrees = sorted(range(len(degrees)), key=lambda i: degrees[i], reverse=True)
    new_net = Network(num_nodes)
    random_node_list = network.node_list()
    np.random.shuffle(random_node_list)
    for i in random_node_list:
        attempt = 0
        while degrees[i] > 0 and attempt < max_attempts:
            degree = degrees[i]
            sorted_index = sorted_degrees.index(i)
            sampled = np.random.normal(loc=sorted_index, scale=std_dev)
            if sampled < 0 or sampled >= len(sorted_degrees):
                sampled_index = np.random.randint(0, len(sorted_degrees))
            else:
                sampled_index = int(math.floor(sampled))
            target_node = sorted_degrees[sampled_index]
            if target_node == i or new_net.is_connected(i, target_node) or degrees[target_node] == 0:
                attempt += 1
                continue
            else:
                new_net.add_edge(i, target_node)
                degrees[i] -= 1
                degrees[target_node] -= 1
                if degrees[target_node] == 0:
                    sorted_degrees.remove(target_node)
                if degrees[i] == 0:
                    sorted_degrees.remove(i)
                    break
    return new_net, degrees


def assortative_negative_network(network):
    degrees = network.degrees()
    num_nodes = network.num_nodes
    sorted_degrees = sorted(range(len(degrees)), key=lambda i: degrees[i], reverse=True)
    new_net = Network(num_nodes)
    for idx_i, i in enumerate(sorted_degrees):
        degree = degrees[i]
        num_connections = 0
        if degree == 0:
            continue
        for idx_j in range(0, len(sorted_degrees) - idx_i - 1):
            j = sorted_degrees[len(sorted_degrees) - idx_j - 1]
            if i == j or degrees[j] == 0:
                continue
            new_net.add_edge(i, j)
            degrees[i] -= 1
            degrees[j] -= 1
            num_connections += 1
            if num_connections == degree or degrees[i] == 0:
                break
    return new_net, degrees
