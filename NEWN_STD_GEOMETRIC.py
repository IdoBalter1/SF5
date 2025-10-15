
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
print("Script to plot mean cluster size and normalized std dev vs. lambda, with std_devs on same plot per network type.")
plt.style.use('ggplot')

# --- Configuration ---
# Define the standard deviation values to iterate over
std_values_to_process = np.array([10, 110, 750, 10000])
# std_values_to_process = np.array([10]) # For faster testing

# Lambda values for the x-axis
lambda_values1 = np.linspace(0, 0.2, 200)
lambda_values2 = np.linspace(0.2,1,100) # Avoid lambda=0; 10 points for faster runs, adjust as needed.
lambda_values = np.concatenate((lambda_values1, lambda_values2))
# Network configurations (Power Law networks with different alpha values)
# The 'color' key here is not used for line colors in this version, as colors distinguish std_devs
network_configs = {
    #'Power Law (α=1.5)': {'creator': lambda: Network.from_power_law(1.5, 10000, 1, 100), 'color_original': 'red'},
    'Geometric (p=1/21)': {'creator': lambda: Network.from_geometric(1/21,10000), 'color_original': 'purple'},
    #'Power Law (α=2.5)': {'creator': lambda: Network.from_power_law(2.5, 10000, 1, 100), 'color_original': 'orange'}
}

# Colors for different std_dev lines
std_dev_colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black'] 


max_attempts_assort_mod = 1000
num_avg_runs_fds = 100 # Number of times to run estimate_outbreak_disjointset_from_network for averaging

# --- Main Loop ---
for network_name, net_config_details in network_configs.items():
    print(f"\n--- Processing for Network Type: {network_name} ---")

    # Create a new figure for this network type
    fig, axs = plt.subplots(1, 2, figsize=(18, 7))
    legend_elements_for_fig = []
    network_template_creator = net_config_details['creator']

    for i, std_dev in enumerate(std_values_to_process):
        print(f"  Processing std_dev: {std_dev:.2f}")
        network_template = network_template_creator() # Create a fresh template for each std_dev modification
        current_std_color = std_dev_colors[i % len(std_dev_colors)]

        mean_sizes_vs_lambda = []
        normalized_stds_vs_lambda = []

        for lam in lambda_values:
            print(f"    λ = {lam:.3f}", end='\r')
            
            net_pos, _, r, _, _ = assortative_network_sample_main(
                network_template, 
                std_dev, 
                max_attempts_assort_mod, 
                'positive'
            )

            if net_pos is not None:
                current_run_means = []
                current_run_stds = []
                for _ in range(num_avg_runs_fds): 
                    sizes = estimate_outbreak_disjointset_from_network(net_pos, lam)
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
            else:
                print(f"Failed to create positively modified network for {network_name} with r={r:.3f} at lambda={lam}")
                mean_sizes_vs_lambda.append(np.nan) 
                normalized_stds_vs_lambda.append(np.nan)
        print(" " * 20, end='\r') 

        # Plot lines for the current std_dev
        line_label = f'r={r:.2f}'
        axs[0].plot(lambda_values, mean_sizes_vs_lambda, color=current_std_color, marker='o', linestyle='-', markersize=4, label=line_label)
        axs[1].plot(lambda_values, normalized_stds_vs_lambda, color=current_std_color, marker='x', linestyle='--', markersize=4, label=line_label)
        
        legend_elements_for_fig.append(Line2D([0], [0], color=current_std_color, lw=2, label=line_label))

    # --- Configure and save the plot for the current network_type ---
    fig.suptitle(f'Cluster Size Analysis vs. Lambda for {network_name}', fontsize=16)

    axs[0].set_xlabel('Lambda (λ)')
    axs[0].set_ylabel('Mean Cluster Size')
    axs[0].set_title('Mean Cluster Size vs. Lambda')
    axs[0].grid(True)

    axs[1].set_xlabel('Lambda (λ)')
    axs[1].set_ylabel('Std. Dev / Mean Cluster Size')
    axs[1].set_title('Normalized Std. Dev. of Cluster Size vs. Lambda')
    axs[1].grid(True)

    fig.legend(handles=legend_elements_for_fig, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=len(std_values_to_process))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.92]) 
    
    # Sanitize network_name for filename
    safe_network_name = network_name.replace(' ', '_').replace('(', '').replace(')', '').replace('α=', 'alpha_').replace('.', 'p')
    save_filename = f"NEWNcluster_analysis_{safe_network_name}_vs_lambda_multi_std2.svg"
    plt.savefig(save_filename)
    print(f"NEWNPlot saved as {save_filename}")
    plt.close(fig)

print("\nScript finished.")