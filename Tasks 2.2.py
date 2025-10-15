from Main import Network, naive_create_network,  gnp_two_stage,dfs,configuration_model, get_degree_distribution
import numpy as np
import matplotlib.pyplot as plt
import random
from itertools import combinations
from collections import defaultdict
from matplotlib import pyplot as plt
import math
import timeit
import networkx as nx
"""
Generate a random number between 0 and 9999. This is the node that we will evaluate.
initilaise an empty list for the amount of neighbours.
Initialise another emptty list for the amount of neighbours of the random neighbour.
Find the neighboours of this node.
add the amount of neighbours to the list.
choose a ranodm neighbour from the list of neighbours.
find the amount of neighbours of this neighbour.
add the amkount of neighbours to the list of the random neighbour.
repeat and average
"""
#Result: Average degree of random node is 10.06, average degree of friend is 11.05 for 1000000 trials.
import random
import numpy as np
import matplotlib.pyplot as plt


"""num_nodes = 10000
lam = 10
geometric_network = Network.from_poisson(lam=lam, num_nodes=num_nodes)


samples = 1000000
n = 10000

random_self = np.random.randint(0, n, size=samples)

geom_self_k_array = []
geom_friend_k_array = []

for i in random_self:
    g_i = i

    # Resample until we find a node with at least one neighbor
    while not len(geometric_network.neighbors(g_i)):
        geom_self_k_array.append(0)  # track that we hit an isolated node
        g_i = np.random.randint(0, n)

    # Get neighbors of this non-isolated node
    g_neighbors = list(geometric_network.neighbors(g_i))

    # Append self degree
    geom_self_k_array.append(len(g_neighbors))

    # Choose a random friend and append their degree
    friend = np.random.choice(g_neighbors)
    geom_friend_k_array.append(len(geometric_network.neighbors(friend)))

# After sampling, you can compute averages
print(f"Average degree of random nodes: {np.mean(geom_self_k_array):.2f}")
print(f"Average degree of their friends: {np.mean(geom_friend_k_array):.2f}")

plt.text(
    0.95, 0.95,
    f"Avg degree (random): {np.mean(geom_self_k_array):.2f}\nAvg degree (friend): {np.mean(geom_friend_k_array):.2f}",
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment='top',
    horizontalalignment='right',
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
)
plt.hist(geom_self_k_array, bins=30, alpha=0.6, label='Random Nodes')
plt.hist(geom_friend_k_array, bins=30, alpha=0.6, label='Friends')
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.title('Friendship Paradox – Poisson Graph')
plt.legend()
plt.show()"""
def convert_to_network(new_net):
    G = nx.Graph()
    for i in new_net.adj:
        for j in new_net.neighbors(i):
            if i<j:
                G.add_edge(i,j)
    return G

def assortative_network_sample(network, std_dev, max_attempts,value):
    a_counter = 0
    b_counter = 0
    degrees = network.degrees()
    num_nodes = network.num_nodes
    sorted_degrees = sorted(range(len(degrees)), key=lambda i: degrees[i], reverse=True) 
    sorted_degrees_reverse = sorted(range(len(degrees)), key=lambda i: degrees[i], reverse=False)  #sorts nodes from lowest to highest degree
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
        #scaled_std_dev = std_dev* len(sorted_degrees) / 10000
        #scaled_std_dev = max(scaled_std_dev, 7)  # Ensure std_dev is at least 1 to avoid too small sampling range
        #print(f"the length of the list is :{len(sorted_degrees)}")
        while degrees[i] > 0 and attempt <max_attempts:
            sorted_index = sorted_degrees.index(i)  # Find the index of the random node in the sorted list
            while attempt < max_attempts:
                sampled = np.random.normal(loc=sorted_index, scale=std_dev) # sampling around the sorted_degrees index
                if sampled >= 0 and sampled < len(sorted_degrees):
                    break
                else:
                    attempt += 1
                    #print(f"this is the :{attempt}st attempt")
            if attempt >= max_attempts:  
                sampled_index = np.random.randint(0, len(sorted_degrees))
                a_counter += 1
            else:
                sampled_index = int(math.floor(sampled))
                b_counter +=1 


            if value == 'positive':
                target_node = sorted_degrees[sampled_index]
            elif value == 'negative':
                target_node = sorted_degrees_reverse[sampled_index]  
            if target_node == i or new_net.is_connected(i, target_node) or degrees[target_node] == 0:
                attempt += 1
                continue
            
            else:
                new_net.add_edge(i, target_node)
                degrees[i] -= 1
                degrees[target_node] -= 1
                if degrees[target_node] ==0 :
                    sorted_degrees.remove(target_node)
                    sorted_degrees_reverse.remove(target_node)
                if degrees[i] == 0:
                    sorted_degrees.remove(i) 
                    sorted_degrees_reverse.remove(i) # Remove the node from the sorted list if its degree is 0
                    break
    G = convert_to_network(new_net)
    r = nx.degree_pearson_correlation_coefficient(G)
    c = nx.average_clustering(G)
    return new_net, degrees, r, a_counter, b_counter,c

network = Network.from_poisson(20, 10000)
std_values = np.linspace(10,8000,10)
for std in std_values:
    new_net, degrees, r,a_counter,b_counter, c = assortative_network_sample(network,std, 1000, value = 'positive' )
    print("Assortative Network Clustering Coefficient:", c)
    print("Assortativity coeffiient:", r)


# print number of degrees 
for i in range (10):
    print(f"Number of nodes with degree {i}:", sum(1 for d in degrees if d == i))
print("Assortative Network Pearson Correlation Coefficient:", r)
print("a counter:", a_counter)
print("b counter:", b_counter)
print("Assortative Network Clustering Coefficient:", c)
print("Assortativity coeffiient:", r)

#print(a_counter*100/(b_counter + a_counter)) # 0.33% we sample randomly from positive assortativity,  