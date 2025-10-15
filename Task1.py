import numpy as np
import random
from itertools import combinations
from collections import defaultdict
from matplotlib import pyplot as plt
import math
#1)
class FastNetwork:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self . adj = { i :set () for i in range ( num_nodes ) }
        self.edge_count = 0

    def add_edge(self, i, j):
        self.adj[i].add(j)
        self.adj[j].add(i)
        self.edge_count += 1

    def neighbors ( self , i ) :
        return self . adj [ i ]

    def edge_list ( self ) :
        return [( i , j ) for i in self . adj for j in self . adj [ i ] if i < j ]
    
    def adjacency_matrix(self):
        mat = np.zeros((self.num_nodes, self.num_nodes), dtype=int)
        for i in self.adj:
            for j in self.adj[i]:
                mat[i, j] = 1
        return mat
    

def naive_create_network(num_nodes, p):
    network = FastNetwork(num_nodes)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.uniform(0,1) <= p:
                network.add_edge(i, j)
    return network, network.edge_count,network.adjacency_matrix()

network, edge_count, adjacency_matrix = naive_create_network(3,0.5)

net = FastNetwork(5)
net.add_edge(0, 1)
net.add_edge(0, 2)
net.add_edge(1, 2)
net.add_edge(1, 3)
net.add_edge(2, 3)
net.add_edge(3, 4)
print(net.adjacency_matrix())



def sample_create__network(num_nodes, p):
    num_edges = int(num_nodes * (num_nodes - 1) / 2)
    m =np.random.binomial(num_edges,(p))
    network = FastNetwork(num_nodes)
    possible_edges = list(combinations(range(num_nodes), 2))
    sampled_edges = random.sample(possible_edges, m)

    for i, j in sampled_edges:
        network.add_edge(i, j)
    
    return network,edge_count ,adjacency_matrix

def calculate_total_pairs(num_nodes):
    """
    Calculate the total number of unique pairs (off-diagonal pairs) for a given number of nodes.
    """
    return num_nodes * (num_nodes - 1) // 2

def gnp_two_stage(num_nodes, p):
    """
    Return a FastNetwork sampled from G(n,p) via the two-stage method:
        1) draw m ~ Binomial(N, p)  where N = n(n-1)/2
        2) choose exactly m edges uniformly at random
    """
    N = calculate_total_pairs(num_nodes)                 # total off-diagonal pairs
    m = np.random.binomial(N, p)         # stage-1: how many edges?

    # stage-2: pick m unique positions in the upper triangle
    rows, cols = np.triu_indices(num_nodes, k=1)         # each length N
    idx = np.random.choice(N, m, replace=False)

    net = FastNetwork(num_nodes)
    for k in idx:                                   # O(m)
        i, j = rows[k], cols[k]
        net.add_edge(i, j)

    return net, edge_count, adjacency_matrix

def dfs(graph, start):
    visited, stack = [], [start] # ✅ Use list
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.append(vertex)
            stack.extend(graph[vertex] - set(visited)) # ✅ Convert to set()
    return visited

def random_graph(num_nodes, p) ->np.ndarray:
    #generate a random graph with probability of edge creation p
    
    adjency_matrix = np.random.binomial(1, p, size = (num_nodes, num_nodes))

    #create a symmetric matrix
    adjency_matrix = np.triu(adjency_matrix) + np.triu(adjency_matrix, 1).T
    np.fill_diagonal(adjency_matrix, 0)
    edge_count = np.sum(np.triu(adjency_matrix, 1))
    return adjency_matrix, edge_count