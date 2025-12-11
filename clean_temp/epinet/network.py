import numpy as np
import random


class Network:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.adj = {i: set() for i in range(num_nodes)}
        self.edge_count = 0

    def add_edge(self, i, j):
        if j not in self.adj[i]:
            self.adj[i].add(j)
            self.adj[j].add(i)
            self.edge_count += 1

    def neighbors(self, i):
        return self.adj[i]

    def edge_list(self):
        return [(i, j) for i in self.adj for j in self.adj[i] if i < j]

    def node_list(self):
        return list(self.adj.keys())

    def adjacency_matrix(self):
        mat = np.zeros((self.num_nodes, self.num_nodes), dtype=int)
        for i in self.adj:
            for j in self.adj[i]:
                mat[i, j] = 1
        return mat

    def degree(self, i):
        return len(self.adj[i])

    def degrees(self):
        return [len(self.adj[i]) for i in range(self.num_nodes)]

    def is_connected(self, i, j):
        return j in self.adj.get(i, set())

    @classmethod
    def from_degree_sequence(cls, k, num_nodes):
        edges = configuration_model(k, num_nodes)
        net = cls(num_nodes)
        for u, v in edges:
            net.add_edge(u, v)
        return net

    @classmethod
    def from_poisson(cls, lam, num_nodes):
        k = np.random.poisson(lam, num_nodes)
        return cls.from_degree_sequence(k, num_nodes)

    @classmethod
    def from_networkx_to_custom(cls, G_nx):
        n = G_nx.number_of_nodes()
        new_net = Network(n)
        for u, v in G_nx.edges():
            new_net.add_edge(u, v)
        return new_net

    @classmethod
    def from_geometric(cls, p, num_nodes):
        k = np.random.geometric(p, num_nodes) - 1
        return cls.from_degree_sequence(k, num_nodes)

    @classmethod
    def from_power_law(cls, alpha, num_nodes, min_deg, max_deg):
        degree_sequence = []
        while len(degree_sequence) < num_nodes:
            k = np.random.zipf(alpha)
            if min_deg <= k <= max_deg:
                degree_sequence.append(k)
        if sum(degree_sequence) % 2 == 1:
            degree_sequence[0] += 1
        return cls.from_degree_sequence(degree_sequence, num_nodes)


def configuration_model(k, num_nodes):
    S = np.array([i for i in range(num_nodes) for _ in range(k[i])])
    S = np.random.permutation(S)
    if len(S) % 2:
        S = S[:-1]
    S = S.reshape(-1, 2)
    return S


def get_degree_distribution(edge_list, num_nodes):
    degree_count = [0] * num_nodes
    for u, v in edge_list:
        degree_count[u] += 1
        degree_count[v] += 1
    return degree_count


def naive_create_network(num_nodes, p):
    network = Network(num_nodes)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.uniform(0, 1) <= p:
                network.add_edge(i, j)
    return network.edge_count, network.adjacency_matrix()


def gnp_two_stage(num_nodes, p):
    N = num_nodes * (num_nodes - 1) // 2
    m = np.random.binomial(N, p)

    net = Network(num_nodes)
    seen = set()

    while len(seen) < m:
        i = random.randint(0, num_nodes - 1)
        j = random.randint(0, num_nodes - 1)
        if i == j:
            continue
        edge = (min(i, j), max(i, j))
        if edge not in seen:
            net.add_edge(i, j)
            seen.add(edge)

    return net, net.edge_count, net.adjacency_matrix()
