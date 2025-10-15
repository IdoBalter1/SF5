import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
from network import Network
from epidemic import length_of_infection_period, simulate_epidemic

plt.style.use('ggplot')


def convert_to_networkx(new_net):
    G = nx.Graph()
    for i in new_net.adj:
        for j in new_net.neighbors(i):
            if i < j:
                G.add_edge(i, j)
    return G


def infection_length_vs_clustering_ws(lambda_val, num_nodes, k, num_weeks, trials, initial_infected):
    p_values_00 = np.linspace(0, 0.01, 8)
    p_values0 = np.linspace(0.012, 0.2, 30)
    p_values1 = np.linspace(0.21, 0.6, 20)
    p_values = np.concatenate((p_values_00, p_values0, p_values1))
    clustering_values = []
    infection_lengths = []
    I_counts = []

    for p in p_values:
        lengths = []
        cluster_coeffs = []
        mean_size = []
        G = nx.watts_strogatz_graph(n=num_nodes, k=k, p=p)
        net = Network.from_networkx_to_custom(G)
        clustering = nx.average_clustering(G)
        r = nx.degree_pearson_correlation_coefficient(G)
        print(f"Assortativity Coefficient for p={p:.3f}: {r:.4f}")
        print(f"Clustering Coefficient for p={p:.3f}: {clustering:.4f}")
        for _ in range(trials):
            cluster_coeffs.append(clustering)
            length = length_of_infection_period(net, initial_infected, lambda_val, num_weeks)
            lengths.append(length)
            _, I_count, _ = simulate_epidemic(net, initial_infected, lambda_val, num_weeks)
            peak_size = max(I_count)
            mean_size.append(peak_size)

        I_counts.append(np.mean(mean_size))
        clustering_values.append(np.mean(cluster_coeffs))
        infection_lengths.append(np.mean(lengths))

    clustering_values_np = np.array(clustering_values)
    infection_lengths_np = np.array(infection_lengths)
    I_counts_np = np.array(I_counts)

    plt.figure(figsize=(10, 6))
    plt.scatter(clustering_values_np, infection_lengths_np, label=f'Infection Length (λ={lambda_val:.2f})', alpha=0.7)
    plt.xlabel('Clustering Coefficient')
    plt.ylabel('Average Length of Infection Period (Weeks)')
    plt.title(f'Infection Length vs. Clustering Coefficient (λ={lambda_val:.2f}, one seed)')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(clustering_values_np, I_counts_np, label=f'Peak Infection Size (λ={lambda_val:.2f})', alpha=0.7)
    plt.xlabel('Clustering Coefficient')
    plt.ylabel('Peak Infection Size')
    plt.title(f'Peak Infection Size vs. Clustering Coefficient (λ={lambda_val:.2f}, one seed)')
    plt.legend()
    plt.grid(True)
    plt.show()
