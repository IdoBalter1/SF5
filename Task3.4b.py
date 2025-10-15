from Main import Network, naive_create_network,fast_disjoint_set,assortative_network_sample_main,convert_to_network, gnp_two_stage, dfs, configuration_model, get_degree_distribution,simulate_epidemic,estimate_outbreak_disjointset_from_network
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from collections import defaultdict
from itertools import combinations
import math
import timeit
import networkx as nx

print("Tswewewt.4: Assortatsdfsdfive Network Sample Main wsdfsdfith Averaging")
network = Network.from_power_law(2.3, 10000,1,100)
G1 = convert_to_network(network)
r_original = nx.degree_pearson_correlation_coefficient(G1) # Get r for the original network
c_original = nx.average_clustering(G1) # Get c for the original network
p_critical = 0.05
lambda_values = np.linspace(0, 1, 100) # Reduced points for faster testing, adjust as needed
std_values = [100,500,1000,2000,5000] 
num_avg_runs = 50 # Number of runs to average over

plt.figure(figsize=(10, 6)) # Adjusted figure size for better legend visibility

# Process and plot the original network with averaging
print(f"Processing original network (r={r_original:.3f}, c={c_original:.3f}) with {num_avg_runs} averaging runs...")
x_orig = random.randint(0, network.num_nodes - 1 if network.num_nodes > 0 else 0)
all_sizes_orig_runs = []
for i in range(num_avg_runs):
    print(f"  Original network, run {i+1}/{num_avg_runs}")
    sizes_run = fast_disjoint_set(network, lambda_values, x_orig)
    all_sizes_orig_runs.append(sizes_run)

if all_sizes_orig_runs:
    averaged_sizes_orig = np.mean(np.array(all_sizes_orig_runs), axis=0)
    if r_original >0:
        plt.plot(lambda_values, averaged_sizes_orig, label=f'Original Network (r={r_original:.3f}) - Avg over {num_avg_runs} runs', marker='s', alpha=0.8, linewidth=1.5)
else:
    print("Warning: No data collected for the original network.")


for std_idx, std in enumerate(std_values):
    print(f"\nProcessing for std_dev: {std} ({std_idx+1}/{len(std_values)}) with {num_avg_runs} averaging runs...")
    # Create the modified network
    new_network, _, r_mod, c_mod, G_mod = assortative_network_sample_main(network, std, 1000, 'positive')
    
    # Randomly select a node index for the disjoint set for this modified network
    x_mod = random.randint(0, new_network.num_nodes - 1 if new_network.num_nodes > 0 else 0)
    
    all_sizes_modified_runs = []
    for i in range(num_avg_runs):
        print(f"  std={std}, run {i+1}/{num_avg_runs}")
        sizes_run = fast_disjoint_set(new_network, lambda_values, x_mod)
        all_sizes_modified_runs.append(sizes_run)

    if all_sizes_modified_runs:
        averaged_sizes_modified = np.mean(np.array(all_sizes_modified_runs), axis=0)
        if r_mod > 0:
            plt.plot(lambda_values, averaged_sizes_modified, label=f'Modified Network (std={std}, r={r_mod:.3f}) - Avg over {num_avg_runs} runs', marker='o', alpha=0.7, linestyle='--')
    else:
        print(f"Warning: No data collected for std_dev={std}.")


plt.xlabel('Lambda Values')
plt.ylabel('Average Cluster Sizes')
plt.title(f'Average Cluster Sizes vs Lambda Values (Averaged over {num_avg_runs} runs)')
plt.legend(fontsize='small')
plt.grid(True)
plt.tight_layout() # Adjust layout to prevent legend overlap
plt.savefig('average_cluster_sizes_vs_lambda2.png', dpi=300) # Save the figure
plt.show()
print("Processing complete.")