import numpy as np
import random
from itertools import combinations
from collections import defaultdict
from matplotlib import pyplot as plt
import math
from scipy.cluster.hierarchy import DisjointSet
from scipy.stats import binom
import networkx as nx
from Main import Network, naive_create_network, gnp_two_stage,estimate_outbreak_disjointset_from_network, dfs, configuration_model, get_degree_distribution, assortative_negative_network,assortative_network,assortative_network_sample,assortative_network_sample_main,convert_to_network,fast_disjoint_set # Ensure Network.from_networkx_to_custom is available if not already
from matplotlib.lines import Line2D # Import Line2D for custom legend

print("Script to plot mean cluster size and normalized std dev vs. lambda for different Power Law alphas (no assortativity mod).")
plt.style.use('ggplot')

# --- Configuration ---
# Lambda values for the x-axis
lambda_values1 = np.linspace(0.01, 0.2, 100) # Start from 0.01 to avoid issues with lambda=0
lambda_values2 = np.linspace(0.201, 1, 100) 
lambda_values = np.concatenate((lambda_values1, lambda_values2))
lambda_values = np.unique(lambda_values) # Ensure unique, sorted values

# Network configurations (Power Law networks with different alpha values)
network_configs = {
    'Power Law (α=1.5)': {'creator': lambda: Network.from_power_law(1.5, 10000, 1, 100), 'color': 'red'},
    'Power Law (α=2.0)': {'creator': lambda: Network.from_power_law(2.0, 10000, 1, 100), 'color': 'purple'},
    'Power Law (α=2.5)': {'creator': lambda: Network.from_power_law(2.5, 10000, 1, 100), 'color': 'orange'}
}

num_avg_runs_fds = 4 # Number of times to run estimate_outbreak_disjointset_from_network for averaging

# --- Main Plotting ---
# Create a single figure for all network types
fig, axs = plt.subplots(1, 2, figsize=(18, 7))
legend_elements_for_fig = []

for network_name, config in network_configs.items():
    print(f"\n--- Processing for Network Type: {network_name} ---")
    
    network_creator = config['creator']
    current_color = config['color']
    
    mean_sizes_vs_lambda = []
    normalized_stds_vs_lambda = []

    # Create the base network once for this alpha value
    base_network = network_creator()

    if base_network is None:
        print(f"  Failed to create base network for {network_name}. Skipping.")
        # Fill with NaNs if network creation fails, so plotting doesn't break
        mean_sizes_vs_lambda = [np.nan] * len(lambda_values)
        normalized_stds_vs_lambda = [np.nan] * len(lambda_values)
    else:
        for lam in lambda_values:
            print(f"    λ = {lam:.3f}", end='\r')
            
            # Use the base_network directly (no assortativity modification)
            # net_pos is the base_network itself
            
            current_run_means = []
            current_run_stds = []
            for _ in range(num_avg_runs_fds): 
                sizes = estimate_outbreak_disjointset_from_network(base_network, lam) # Use base_network
                if sizes: 
                    current_run_means.append(np.mean(sizes))
                    current_run_stds.append(np.std(sizes))
                else: 
                    current_run_means.append(0)
                    current_run_stds.append(0)
            
            avg_mean_size_for_lam = np.mean(current_run_means) if current_run_means else 0
            avg_std_for_lam = np.mean(current_run_stds) if current_run_stds else 0

            mean_sizes_vs_lambda.append(avg_mean_size_for_lam)
            if avg_mean_size_for_lam > 0:
                normalized_stds_vs_lambda.append(avg_std_for_lam / avg_mean_size_for_lam)
            else:
                normalized_stds_vs_lambda.append(0) 
        print(" " * 20, end='\r') 

    # Plot lines for the current network type (alpha value)
    line_label = network_name # Use the descriptive network name for the legend
    axs[0].plot(lambda_values, mean_sizes_vs_lambda, color=current_color, marker='o', linestyle='-', markersize=3, label=line_label)
    axs[1].plot(lambda_values, normalized_stds_vs_lambda, color=current_color, marker='x', linestyle='--', markersize=3, label=line_label)
    
    legend_elements_for_fig.append(Line2D([0], [0], color=current_color, lw=2, label=line_label))

# --- Configure and save the single plot ---
fig.suptitle('Cluster Size Analysis vs. Lambda for Different Power Law Alphas (Base Networks)', fontsize=16)

axs[0].set_xlabel('Lambda (λ)')
axs[0].set_ylabel('Mean Cluster Size')
axs[0].set_title('Mean Cluster Size vs. Lambda')
axs[0].grid(True)
# axs[0].legend() # Individual legends per subplot can be noisy; using a figure-level legend

axs[1].set_xlabel('Lambda (λ)')
axs[1].set_ylabel('Std. Dev / Mean Cluster Size')
axs[1].set_title('Normalized Std. Dev. of Cluster Size vs. Lambda')
axs[1].grid(True)
# axs[1].legend()

fig.legend(handles=legend_elements_for_fig, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=len(network_configs))

plt.tight_layout(rect=[0, 0.03, 1, 0.92]) 

save_filename = "cluster_analysis_power_law_alphas_vs_lambda.svg"
plt.savefig(save_filename)
print(f"\nPlot saved as {save_filename}")
plt.close(fig)

print("\nScript finished.")
