from Main import Network,assortative_network,assortative_network_sample, assortative_negative_network, naive_create_network, gnp_two_stage, dfs, configuration_model, get_degree_distribution
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, geom
import random
import time
# Generate a poisson graph with mean between 0 and 2
#create a list of average degrees between 0 and 2
"""num_nodes = 2**10
average_degrees = np.linspace(0, 2, 50)
print("negative assortative network")
average_degree_list_poisson = []
average_degree_list_geometric = []
average_degree_list_poisson_biased = []
average_degree_list_geometric_biased = []
for average_degree in average_degrees:
    lam = average_degree
    p = 1/(average_degree + 1)
    print(f"Average Degree: {average_degree}, Poisson Lambda: {lam}, Geometric p: {p}")  
    net1 = Network.from_poisson(lam, num_nodes)
    net2 = Network.from_geometric(p, num_nodes)
    net3 = assortative_negative_network(net1)
    net4 = assortative_negative_network(net2)
    visited_counts_poisson = []
    visited_biased_poisson = []
    visited_biased_geometric = []
    visited_counts_geometric = []
    for _ in range(10):
        x = np.random.randint(0, num_nodes)
        print(x)
        for i in range(1000):
            visited_poisson = dfs(net1.adj, x)
            visited_counts_poisson.append(len(visited_poisson))
            visited_geometric = dfs(net2.adj, x)
            visited_counts_geometric.append(len(visited_geometric))
            visited_biased_poisson = dfs(net3.adj, x)
            visited_biased_poisson.append(len(visited_biased_poisson))
            visited_biased_geometric = dfs(net4.adj, x)
            visited_biased_geometric.append(len(visited_biased_geometric))


    average_degree_list_poisson.append(np.mean(visited_counts_poisson))
    average_degree_list_geometric.append(np.mean(visited_counts_geometric))
    average_degree_list_poisson_biased.append(np.mean(visited_biased_poisson))
    average_degree_list_geometric_biased.append(np.mean(visited_biased_geometric))

# Plot the results
plt.figure(figsize=(12, 6))     
plt.plot(average_degrees, average_degree_list_poisson, marker='o', label='Poisson Degree Distribution')
plt.plot(average_degrees, average_degree_list_geometric, marker='o', label='Geometric Degree Distribution')
plt.plot(average_degrees, average_degree_list_poisson_biased, marker='x', label='Negative Assortative Poisson Degree Distribution')
plt.plot(average_degrees, average_degree_list_geometric_biased, marker='x', label='Negative Assortative Geometric Degree Distribution')
plt.axvline(x = 0.5, color='r', linestyle='--', label='Critical Average Degree for geometric distribution (0.5)')
plt.axvline(x = 1, color='g', linestyle='--', label='Critical Average Degree for Poisson distribution (1)')
plt.xlabel('Average Degree')
plt.ylabel('Average Size of Component Containing Node')
plt.title('Component Size vs Average Degree for Poisson and Geometric Distributions')
plt.legend(fontsize='small')
plt.grid(True)
plt.show()"""


import numpy as np
import matplotlib.pyplot as plt
import random



num_nodes = 10000
samples = 100000

p = 10
net= Network.from_poisson(p, num_nodes=num_nodes)
for std in range(3,10,1):
    new_net,_ = assortative_network_sample(net, std, 100,'negative')  # Biased (e.g. negatively assortative) network

    # Arrays to store degree data
    geom_self_k_array = []
    geom_friend_k_array = []
    geom_self_k_biased = []
    geom_friend_k_biased = []

    # Pre-sample nodes
    random_nodes = np.random.randint(0, num_nodes, size=samples)

    for node_id in random_nodes:
        # --- Unbiased network ---
        unbiased_node = node_id
        while net.degree(unbiased_node) == 0:
            geom_self_k_array.append(0)  # Include 0 for isolated node
            unbiased_node = np.random.randint(0, num_nodes)

        unbiased_neighbors = list(net.neighbors(unbiased_node))
        geom_self_k_array.append(net.degree(unbiased_node))
        unbiased_friend = np.random.choice(unbiased_neighbors)
        geom_friend_k_array.append(net.degree(unbiased_friend))

        # --- Biased network ---
        biased_node = node_id
        while new_net.degree(biased_node) == 0:
            geom_self_k_biased.append(0)
            biased_node = np.random.randint(0, num_nodes)

        biased_neighbors = list(new_net.neighbors(biased_node))
        geom_self_k_biased.append(new_net.degree(biased_node))
        biased_friend = np.random.choice(biased_neighbors)
        geom_friend_k_biased.append(new_net.degree(biased_friend))
    # --- Results ---
    print(f"Average degree of random nodes: {np.mean(geom_self_k_array):.2f}")
    print(f"Average degree of their friends: {np.mean(geom_friend_k_array):.2f}")
    print(f"Average degree of random nodes (biased): {np.mean(geom_self_k_biased):.2f}")
    print(f"Average degree of their friends (biased): {np.mean(geom_friend_k_biased):.2f}")

    # --- Plotting ---
    plt.figure(figsize=(14, 6))

    # Unbiased
    plt.subplot(1, 2, 1)
    plt.hist(geom_self_k_array, bins=30, alpha=0.6, label='Random Nodes', zorder=1)
    plt.hist(geom_friend_k_array, bins=30, alpha=0.6, label='Friends', zorder=2)
    plt.axvline(np.mean(geom_self_k_array), color='blue', linestyle='--', label=f'Avg Random: {np.mean(geom_self_k_array):.2f}', zorder=3)
    plt.axvline(np.mean(geom_friend_k_array), color='orange', linestyle='--', label=f'Avg Friend: {np.mean(geom_friend_k_array):.2f}', zorder=4)
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Unbiased Poisson Network')
    plt.legend()

    # Biased
    plt.subplot(1, 2, 2)
    plt.hist(geom_self_k_biased, bins=30, alpha=0.6, label='Random Nodes (Biased)', zorder=1)
    plt.hist(geom_friend_k_biased, bins=30, alpha=0.6, label='Friends (Biased)', zorder=2)
    plt.axvline(np.mean(geom_self_k_biased), color='blue', linestyle='--', label=f'Avg Random: {np.mean(geom_self_k_biased):.2f}', zorder=3)
    plt.axvline(np.mean(geom_friend_k_biased), color='orange', linestyle='--', label=f'Avg Friend: {np.mean(geom_friend_k_biased):.2f}', zorder=4)
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Biased Poisson Network')
    plt.legend()

    plt.tight_layout()
    plt.show()