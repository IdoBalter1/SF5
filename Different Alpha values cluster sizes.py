import numpy as np
import random
from itertools import combinations
from collections import defaultdict
from matplotlib import pyplot as plt
import math
from scipy.cluster.hierarchy import DisjointSet
from scipy.stats import binom
import networkx as nx
from Main import Network, naive_create_network, gnp_two_stage,estimate_outbreak_disjointset_from_network, dfs, configuration_model, get_degree_distribution, assortative_negative_network,assortative_network,assortative_network_sample,assortative_network_sample_main,convert_to_network,fast_disjoint_set # Ensure Network.from_networkx_to_custom is available
from matplotlib.lines import Line2D # Import Line2D for custom legend

print("savedpowerlawsdfdfsampledifferent xsdf with Watts-Strogatz")
plt.style.use('ggplot')
print("plt.style.use('ggplot')")

# --- Original target script configurations ---
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

# --- Configuration for Watts-Strogatz (from example) ---
n_nodes_main = 10000
k_neighbors_main = 20 
ws_p_rewire_values1 = np.linspace(0, 0.2, 200) # Reduced number of points for faster execution
ws_p_rewire_values2 = np.linspace(0.201, 0.6, 100) # Reduced number of points
ws_p_rewire_values_main = np.concatenate((ws_p_rewire_values1, ws_p_rewire_values2))
ws_plot_color = 'gold'
ws_marker = '^' 
ws_legend_name = f"Watts-Strogatz (N={n_nodes_main}, k={k_neighbors_main}, various p)"


# --- Modified network_configs (list of dicts) ---
# Base configurations that undergo assortativity modification
network_configs_list = [
    {
        "name": "Poisson", # Internal identifier
        "display_name": f"Poisson (k={k_neighbors_main}, N = 10000)", # For legend
        "network_creator": lambda: Network.from_poisson(k_neighbors_main, n_nodes_main),
        "base_color": 'blue',
        "marker": 'o',
        "apply_assortativity_modification": True
    },
    {
        "name": "Geometric",
        "display_name": f"Geometric (p=1/{(k_neighbors_main+1)},N = 10000)",
        "network_creator": lambda: Network.from_geometric(1/(k_neighbors_main+1), n_nodes_main),
        "base_color": 'green',
        "marker": 's',
        "apply_assortativity_modification": True
    },
    {
        "name": "Power Law",
        "display_name": f"Power Law (α=2, N = 10000)", # Assuming alpha=2 as in original target script
        "network_creator": lambda: Network.from_power_law(2, n_nodes_main, 1, 100),
        "base_color": 'red',
        "marker": 'x',
        "apply_assortativity_modification": True
    }
]

# Add Watts-Strogatz configurations (do not undergo std modification loop)
for p_rewire_val in ws_p_rewire_values_main:
    network_configs_list.append({
        "name": f"Watts-Strogatz p={p_rewire_val:.3f}", # Unique internal name
        "display_name": ws_legend_name, # Common legend name for all WS points
        "network_creator": lambda p_r=p_rewire_val: Network.from_networkx_to_custom(
            nx.watts_strogatz_graph(n=n_nodes_main, k=k_neighbors_main, p=p_r)
        ),
        "base_color": ws_plot_color,
        "marker": ws_marker,
        "apply_assortativity_modification": False,
        "actual_p_rewire": p_rewire_val 
    })

lambda_values = [0.4] # As in original target script

for lam in lambda_values:
    print(f"--- Processing Lambda: {lam:.3f} ---")
    
    fig, axs = plt.subplots(1, 2, figsize=(18, 7))
    all_legend_elements_for_lambda = [] 
    added_legend_labels = set() # To ensure one legend item per display_name

    for config in network_configs_list:
        network_display_name = config["display_name"]
        base_network_template = config["network_creator"]() # Create network object
        current_color = config["base_color"]
        current_marker = config["marker"]

        # Use the unique internal name for detailed print statements
        print(f"  Processing network config: {config['name']} for lambda: {lam:.3f}")
        
        if config.get("apply_assortativity_modification", True):
            # This block is for Poisson, Geometric, Power Law (undergoes std modification)
            for std in std_values: 
                print(f"    Processing std_dev: {std:.2f} for {config['name']}") 
                
                # Positive assortativity
                # Create a fresh base network for each std modification if base_network_template is a template
                # Or, if base_network_template is already a network object, it will be modified.
                # The original target script passed network_template (an object) to assortative_network_sample_main.
                # Let's assume assortative_network_sample_main handles copying or works on a modifiable copy.
                net_pos, _, r_pos, c_pos, _ = assortative_network_sample_main(base_network_template, std, max_attempts_assort_mod, 'positive')
                
                if net_pos is not None:
                    mean_cluster_sizes_pos = []
                    for _ in range(num_avg_runs_fds):
                        sizes = estimate_outbreak_disjointset_from_network(net_pos, lam)
                        mean_cluster_sizes_pos.append(np.mean(sizes) if sizes and len(sizes) > 0 else 0)
                    total_mean_pos = np.mean(mean_cluster_sizes_pos) if mean_cluster_sizes_pos else 0
                    
                    axs[0].scatter(r_pos, total_mean_pos, color=current_color, marker=current_marker, alpha=0.6) 
                    axs[1].scatter(c_pos, total_mean_pos, color=current_color, marker=current_marker, alpha=0.6)

                # Negative assortativity
                process_negative = False
                if (config["name"] == 'Geometric' and std > 100) or \
                   (config["name"] == 'Poisson'): # Match internal names
                    process_negative = True
                
                if process_negative:
                    net_neg, _, r_neg, c_neg, _ = assortative_network_sample_main(base_network_template, std, max_attempts_assort_mod, 'negative')
                    if net_neg is not None:
                        mean_cluster_sizes_neg = []
                        for _ in range(num_avg_runs_fds):
                            sizes_neg = estimate_outbreak_disjointset_from_network(net_neg, lam)
                            mean_cluster_sizes_neg.append(np.mean(sizes_neg) if sizes_neg and len(sizes_neg) > 0 else 0)
                        total_mean_neg = np.mean(mean_cluster_sizes_neg) if mean_cluster_sizes_neg else 0

                        axs[0].scatter(r_neg, total_mean_neg, color=current_color, marker=current_marker, alpha=0.6)
                        axs[1].scatter(c_neg, total_mean_neg, color=current_color, marker=current_marker, alpha=0.6)
        else:
            # This block is for Watts-Strogatz (and any other non-modified networks)
            # base_network_template is the specific WS graph for a p_rewire_val
            # No std loop here.
            g_nx_for_metrics = convert_to_network(base_network_template) # Convert custom Network to NetworkX graph
            r_val = 0.0
            c_val = 0.0
            if g_nx_for_metrics and g_nx_for_metrics.number_of_nodes() > 0 : # Check if graph is not empty
                r_val = nx.degree_pearson_correlation_coefficient(g_nx_for_metrics) if g_nx_for_metrics.number_of_edges() > 0 else 0.0
                c_val = nx.average_clustering(g_nx_for_metrics)

            mean_cluster_sizes_base = []
            for _ in range(num_avg_runs_fds):
                sizes = estimate_outbreak_disjointset_from_network(base_network_template, lam)
                mean_cluster_sizes_base.append(np.mean(sizes) if sizes and len(sizes) > 0 else 0)
            total_mean_base = np.mean(mean_cluster_sizes_base) if mean_cluster_sizes_base else 0
            
            axs[0].scatter(r_val, total_mean_base, color=current_color, marker=current_marker, alpha=0.7)
            axs[1].scatter(c_val, total_mean_base, color=current_color, marker=current_marker, alpha=0.7)

        # Add legend element (Line2D) if this display_name hasn't been added for this lambda plot
        if network_display_name not in added_legend_labels:
            all_legend_elements_for_lambda.append(
                Line2D([0], [0], marker=current_marker, color='w', 
                       label=network_display_name, 
                       markerfacecolor=current_color, markersize=8)
            )
            added_legend_labels.add(network_display_name)

    # --- Configure and save the plot for the current lambda (after all network types) ---
    axs[0].set_xlabel('Assortativity Coefficient (r)')
    axs[0].set_ylabel('Mean Cluster Size')
    axs[0].set_title(f'vs. Assortativity (r)')
    axs[0].grid(True)

    axs[1].set_xlabel('Clustering Coefficient (C_g)')
    axs[1].set_ylabel('Mean Cluster Size')
    axs[1].set_title(f'vs. Clustering (C_g)')
    axs[1].grid(True)
    
    fig.suptitle(f'Mean Cluster Size Analysis for All Networks at λ={lam:.3f}', fontsize=16)
    
    if all_legend_elements_for_lambda: # Only add legend if there are elements
        # Place legend on the bottom left of the first subplot (axs[0])
        axs[0].legend(handles=all_legend_elements_for_lambda, loc='lower left', fontsize='small')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    
    save_filename = f"NEWNplot_all_networks_lambda_{lam:.3f}1_with_WS.svg" # Updated filename
    plt.savefig(save_filename)
    print(f"    Plot saved as {save_filename}")
    plt.close(fig) 

print("Script finished.")
