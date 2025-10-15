from Main import Network, naive_create_network,assortative_network_sample_main,convert_to_network, gnp_two_stage, dfs, configuration_model, get_degree_distribution,simulate_epidemic,estimate_outbreak_disjointset_from_network
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from collections import defaultdict
from itertools import combinations
import math
import timeit
import networkx as nx
# --- Configuration ---
N_nodes_ws = 10000
K_neighbors_ws = 20
# lambda_values should be defined before this script, or define it here:
# lambda_values = np.linspace(0, 0.15, 50) # Example
p_vals_ws = np.linspace(0, 0.6, 5)      # Example: 5 p-values for different lines
num_simulation_runs_avg = 1
# Create figure and subplots once
fig, axs = plt.subplots(1, 2, figsize=(15, 6))
lambda_values = np.linspace(0, 0.15, 2) # Example: 50 lambda values for the x-axis

# --- Main Logic ---
for p_current_ws_graph in p_vals_ws: # Outer loop: each p_val creates a new line on the plots
    print(f"Processing Watts-Strogatz with p = {p_current_ws_graph:.3f}")
    
    # Generate Watts-Strogatz graph for the current p_current_ws_graph
    nx_graph_ws = nx.watts_strogatz_graph(n=N_nodes_ws, k=K_neighbors_ws, p=p_current_ws_graph)
    c = nx.average_clustering(nx_graph_ws)  # Calculate clustering coefficient
    r = nx.degree_pearson_correlation_coefficient(nx_graph_ws)  # Calculate assortativity coefficient
    print(f"  Assortativity Coefficient: {r:.4f}, Clustering Coefficient: {c:.4f}")  # Show progress
    
    # Convert the NetworkX graph to your custom Network object
    custom_network_ws = Network.from_networkx_to_custom(nx_graph_ws)

    # Lists to store results for the current p_current_ws_graph (i.e., for the current line)
    mean_sizes_for_current_p_line = []
    std_over_mean_for_current_p_line = []

    for lam_current_x_axis in lambda_values: # Inner loop: iterates through lambda values for the x-axis
        # print(f"  Lambda: {lam_current_x_axis:.3f}") # Optional: for detailed progress
        
        all_outbreak_sizes_for_this_lambda = []
        for _ in range(num_simulation_runs_avg): # Averaging runs for the current (p, lambda)
            # Use the custom_network_ws generated for the current p_current_ws_graph
            sizes_from_one_sim_run = estimate_outbreak_disjointset_from_network(custom_network_ws, lam_current_x_axis)
            
            # Assuming estimate_outbreak_disjointset_from_network returns a list of sizes or a single size
            if isinstance(sizes_from_one_sim_run, (list, np.ndarray)):
                all_outbreak_sizes_for_this_lambda.extend(sizes_from_one_sim_run)
            else:
                all_outbreak_sizes_for_this_lambda.append(sizes_from_one_sim_run)
        
        current_mean_for_lambda = 0
        current_std_for_lambda = 0
        if all_outbreak_sizes_for_this_lambda: # Avoid division by zero if no sizes
            current_mean_for_lambda = np.mean(all_outbreak_sizes_for_this_lambda)
            current_std_for_lambda = np.std(all_outbreak_sizes_for_this_lambda)
        
        mean_sizes_for_current_p_line.append(current_mean_for_lambda)
        std_over_mean_for_current_p_line.append(current_std_for_lambda / current_mean_for_lambda if current_mean_for_lambda != 0 else 0)

    # Plot the results for the current p_current_ws_graph
    plot_label = f'r = {r:.4f}, c = {c:.4f}'
    axs[0].plot(lambda_values, mean_sizes_for_current_p_line, 'o-', markersize=3, label=plot_label)
    axs[1].plot(lambda_values, std_over_mean_for_current_p_line, 'o-', markersize=3, label=plot_label)

# --- Final Plot Configuration ---
axs[0].set_xlabel('Lambda')
axs[0].set_ylabel('Mean Outbreak Size')
axs[0].set_title('Mean Outbreak Size vs. Lambda (Watts-Strogatz)')
axs[0].legend(loc='upper left', fontsize='small')
axs[0].grid(True)

axs[1].set_xlabel('Lambda')
axs[1].set_ylabel('Std / Mean Outbreak Size')
axs[1].set_title('Std/Mean Outbreak Size vs. Lambda (Watts-Strogatz)')
axs[1].legend(loc='upper right', fontsize='small')
axs[1].grid(True)

plt.tight_layout()
plt.savefig('watts_strogatz_outbreak_size_analysis.png', dpi=300)  # Save the figure