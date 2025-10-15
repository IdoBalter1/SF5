from Main import Network, naive_create_network, gnp_two_stage,dfs
import numpy as np
import random
from itertools import combinations
from collections import defaultdict
from matplotlib import pyplot as plt
import math
import timeit

num_nodes = 1000
p = 0.5  # Example probability value; adjust as needed
net, matrix, _ = gnp_two_stage(num_nodes, p)
#make the code loop 1000 times
delta_list_mean = []
all_deltas = []
for f in range(10):
    net, matrix, _ = gnp_two_stage(num_nodes, p)  # Generate a new network each time
    delta_list = []
    for i in range(num_nodes):
        k_i = net.degree(i)
        if k_i == 0:
            continue
        else:
            neighbor_degrees = [net.degree(x) for x in net.neighbors(i)]
            kappa_i = np.mean(neighbor_degrees)
            delta_i = kappa_i - k_i
            delta_list.append(delta_i)
    if delta_list:  # Avoid division by zero if all nodes are isolated
        delta_list_mean.append(np.mean(delta_list))
        all_deltas.extend(delta_list)

print(delta_list_mean)

assert all(delta > 0 for delta in delta_list_mean), "Not every delta is positive"
plt.text(0.5, 0.95, f"Mean Δi: {np.mean(delta_list_mean):.2f}")
plt.figure(figsize=(8, 6))

plt.axvline(x = np.mean(delta_list_mean), color='r', linestyle='--', label='Mean Δi')
plt.legend()
plt.hist(delta_list, bins=30, alpha=0.7, edgecolor='black')
plt.title("Histogram of Δi = avg(friend degree) − own degree")
plt.xlabel("Δi")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()


def create_sample(maximum_length, maximum_attempts,idx, std):
    sampled = -1
    attempt = 0 
    while sampled < 0  or sampled >= maximum_length and attempt < maximum_attempts:
        sampled = np.random.normal(loc = idx, scale = std)
        sampled_index = int(math.floor(sampled))



def assortative_network_sample(network, std_dev, max_attempts, type):
    degrees = network.degrees()
    num_nodes = network.num_nodes
    sorted_degrees = sorted(range(len(degrees)), key=lambda i: degrees[i], reverse=True) 
    new_net = Network(num_nodes) # create a new network for the new assortative network
    # create a random list of nodes from 0 to num_nodes-1
    random_node_list = network.node_list()
    np.random.shuffle(random_node_list)
    # Find the index of the random node in the sorted list
    # Create a distribution that favours nodes around the random node to be chosen
    # pick a node from the distribution around the random node
    # connect the nodes, and decrease the degree of the nodes
    # continue until the degree of the node is 0 or we have connected all the nodes
    # it might be faster to remove the node from the sorted_degrees list after it has been connected but we can do that after.
    for i in random_node_list:
        attempt = 0
        while degrees[i] > 0 and attempt < max_attempts:
            sorted_index = sorted_degrees.index(i)  # Find the index of the random node in the sorted list
            sampled = -1 
            while sampled < 0 or sampled >= len(sorted_degrees) or  attempt > max_attempts:
                if type == 'positive':
                    sampled = np.random.normal(loc=sorted_index, scale=std_dev)
                    sampled_index = int(math.floor(sampled))
                    attempt += 1
                elif type == 'negative':
                    sampled = np.random.normal(loc = len(sorted_degrees) - sorted_index - 1, scale=std_dev)
                    sampled_index = int(math.floor(sampled))
                    attempt +=1
            else:
                sampled_index = np.random.randint(0, len(sorted_degrees))
                
            
            
            target_node = sorted_degrees[sampled_index]
            if target_node == i or new_net.is_connected(i, target_node) or degrees[target_node] == 0:
                attempt += 1
                continue
            
            else:
                new_net.add_edge(i, target_node)
                degrees[i] -= 1
                degrees[target_node] -= 1
                if degrees[target_node] ==0 :
                    sorted_degrees.remove(target_node)
                if degrees[i] == 0:
                    sorted_degrees.remove(i)  # Remove the node from the sorted list if its degree is 0
                    break

    G = nx.Graph()
    G.add_edges_from(new_net.edge_list())
    r = nx.degree_pearson_correlation_coefficient(G)
    return new_net, degrees, r
   