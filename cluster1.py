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

print("savedpowerlawsdfdfsampledifferent xsdf")
plt.style.use('ggplot')
print("plt.style.use('ggplot')")

# These global variables are used by the function
std_values_low_conf = np.linspace(4.5,20,20)
std_values_low_mid_conf = np.linspace(21,60,40)
std_mid_conf = np.linspace(65,100,30)
std_high_conf = np.linspace(105,300,80)
std_high_conf1 = np.linspace(315,6000,200)
std_high_high_conf = np.linspace(6100,15000,100)
std_values  = np.concatenate((std_values_low_conf,std_values_low_mid_conf, std_mid_conf,std_high_conf,std_high_conf1,std_high_high_conf))
print(f"Total std_values: {len(std_values)}") 
max_attempts_assort_mod = 1000 
num_avg_runs_fds = 1

# network_modification_type = 'positive' # This is handled locally now

network_configs = {
    'poisson': Network.from_poisson(20, 10000),
    'geometric': Network.from_geometric(1/21, 10000),
    'power_law': Network.from_power_law(2, 10000, 1, 100)
}
lambda_values = [0.4]# Using 100 lambda values

for lam in lambda_values:
    print(f"--- Processing Lambda: {lam:.3f} ---")
    
    # Create a figure with two subplots ONCE for this lambda
    fig, axs = plt.subplots(1, 2, figsize=(18, 7))
    all_legend_elements_for_lambda = [] # To store legend handles for all network types for this lambda

    for network_type, network_template in network_configs.items():
        print(f"  Processing network type: {network_type} for lambda: {lam:.3f}")
        
        color_map_for_network = { # Define or access your full color map here if not already global
            'poisson': 'blue',
            'geometric': 'green',
            'power_law': 'red'
            # Add other network types if they exist
        }
        current_color = color_map_for_network[network_type]
        
        for std in std_values: # This loop currently runs only once as std_values = [7]
            print(f"    Processing std_dev: {std:.2f}") 
            
            net_pos, _, r_pos, c_pos, _ = assortative_network_sample_main(network_template, std, max_attempts_assort_mod, 'positive')
            
            net_neg = None
            r_neg, c_neg = None, None # Initialize
            # Condition for attempting negative assortativity
            if (network_type == 'geometric' and std > 100) or network_type =='poisson': 
                net_neg, _, r_neg, c_neg, _ = assortative_network_sample_main(network_template, std, max_attempts_assort_mod, 'negative')
            
            # Positive assortativity
            if net_pos is not None:
                mean_cluster_sizes_pos = []
                for _ in range(num_avg_runs_fds):
                    sizes = estimate_outbreak_disjointset_from_network(net_pos, lam)
                    mean_cluster_sizes_pos.append(np.mean(sizes))
                total_mean_pos = np.mean(mean_cluster_sizes_pos)
                
                axs[0].scatter(r_pos, total_mean_pos, color=current_color, alpha=0.6, 
                               label=f'{network_type} positive (std={std:.2f})') # Keep label for clarity if needed, or remove if legend is fully custom
                axs[1].scatter(c_pos, total_mean_pos, color=current_color, alpha=0.6, 
                               label=f'{network_type} positive (std={std:.2f})')

            # Negative assortativity
            if net_neg is not None:
                mean_cluster_sizes_neg = []
                for _ in range(num_avg_runs_fds):
                    sizes_neg = estimate_outbreak_disjointset_from_network(net_neg, lam)
                    mean_cluster_sizes_neg.append(np.mean(sizes_neg))
                total_mean_neg = np.mean(mean_cluster_sizes_neg)

                axs[0].scatter(r_neg, total_mean_neg, color=current_color, alpha=0.6, # marker='x' removed as per previous request
                               label=f'{network_type} negative (std={std:.2f})')
                axs[1].scatter(c_neg, total_mean_neg, color=current_color, alpha=0.6, # marker='x' removed
                               label=f'{network_type} negative (std={std:.2f})')

        # Add a legend element for the current network_type
        all_legend_elements_for_lambda.append(
            Line2D([0], [0], marker='o', color='w', 
                   label=f'{network_type.capitalize()}', 
                   markerfacecolor=current_color, markersize=8)
        )

    # --- Configure and save the plot for the current lambda (after all network types) ---
    
    # --- Subplot 1: Assortativity ---
    axs[0].set_xlabel('Assortativity Coefficient (r)')
    axs[0].set_ylabel('Mean Cluster Size')
    axs[0].set_title(f'vs. Assortativity (r)')
    axs[0].grid(True)

    # --- Subplot 2: Clustering ---
    axs[1].set_xlabel('Clustering Coefficient (C_g)')
    axs[1].set_ylabel('Mean Cluster Size')
    axs[1].set_title(f'vs. Clustering (C_g)')
    axs[1].grid(True)
    
    # --- Figure Title and Legend ---
    fig.suptitle(f'Mean Cluster Size Analysis for All Networks at λ={lam:.3f}', fontsize=16)
    
    # Place combined legend on the figure
    fig.legend(handles=all_legend_elements_for_lambda, loc='upper right', bbox_to_anchor=(0.99, 0.95))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
    
    plt.savefig(f"LASTONE{lam:.3f}1.svg")
    plt.close(fig) # Close figure after saving

print("Script finished.")




