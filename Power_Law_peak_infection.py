import numpy as np
import random
from itertools import combinations
from collections import defaultdict
from matplotlib import pyplot as plt
import math
from scipy.cluster.hierarchy import DisjointSet
from scipy.stats import binom
import networkx as nx
from Main import Network, convert_to_network, assortative_network_sample_main, simulate_epidemic
print("savesdfdfd")
print("plt.style.use('ggplot')")
print("Task 3.5: Peak Infection Metrics")
plt.style.use('ggplot')  # Optional: Use ggplot style for better aesthetics
# --- Configuration ---
std_values_low_conf = np.linspace(4.5,20,20)
std_values_low_mid_conf = np.linspace(21,60,40)
std_mid_conf = np.linspace(65,100,30)
std_high_conf = np.linspace(105,300,80)
std_high_conf1 = np.linspace(315,6000,200)
std_high_high_conf = np.linspace(6100,15000,100)
std_values_conf  = np.concatenate((std_values_low_conf,std_values_low_mid_conf, std_mid_conf,std_high_conf,std_high_conf1,std_high_high_conf))
print(len(std_values_conf))
max_attempts_assort_config = 1000
num_initial_to_infect_config = 10
num_weeks_simulation_config = 150
num_avg_runs_peak_config = 100 # Number of simulation runs to average peak size

# --- Helper function for networks undergoing assortativity modification ---
def get_network_peak_infection_data_modified(
    base_network_orig,
    std_dev_values_list, # Renamed to avoid conflict with global
    # initial_infected_nodes, # Removed
    lam_val,
    num_weeks_sim,
    num_avg_runs,
    network_label, # Simplified parameter name
    max_attempts_assort,
    num_nodes_to_infect_per_trial # Added
):
    r_points = []
    c_points = []
    peak_infection_points = []

    for std in std_dev_values_list:
        # Skip geometric networks if std_dev <= 7.5 for positive modification (already present)
        if not ("Geometric" in network_label and std <= 7.5):
            print(f"    Skipping {network_label} with std_dev: {std:.2f} (std_dev <= 7.5 for positive path)")
            # If we skip positive, we might also want to skip negative for this std.
            # However, the new condition is specifically for the negative block.
            # For now, this 'continue' only affects the positive part for Geometric std <= 7.5
            # If the intent is to skip *all* processing for this std for Geometric,
            # this continue is fine. If only positive, then negative might still run if its condition is met.
            # Given the prompt, the new condition is *only* for the negative block.
            # The existing skip for Geometric std <= 7.5 will prevent positive data collection.
            # The negative block will then be evaluated independently with its own condition.

            # Positive assortativity
            print(f"    Processing positive assortativity for {network_label} with std_dev: {std:.2f}")
            net_pos, _, r_pos, c_pos, _ = assortative_network_sample_main(base_network_orig, std, max_attempts_assort, 'positive')
            avg_peak_pos_list = []
            # Assuming net_pos is always successfully created and has nodes
            available_nodes_pos = list(net_pos.adj.keys()) if net_pos.adj else list(range(net_pos.num_nodes))
            for _ in range(num_avg_runs):
                current_initial_infected_pos = random.sample(available_nodes_pos, k=num_nodes_to_infect_per_trial)
                _, I_counts, _ = simulate_epidemic(net_pos, current_initial_infected_pos, lam_val, num_weeks_sim)
                avg_peak_pos_list.append(max(I_counts) if I_counts else 0)
            
            r_points.append(r_pos)
            c_points.append(c_pos)
            peak_infection_points.append(np.mean(avg_peak_pos_list))

        # Condition for attempting negative assortativity
        # (network_type == 'geometric' and std > 7.5) or network_type == 'poisson'
        # We infer network_type from network_label
        process_negative_assortativity = False
        if ("Geometric" in network_label and std > 100) or \
           ("Poisson" in network_label):
            process_negative_assortativity = True
        
        # If it is a power law, the above condition will implicitly exclude it unless "Poisson" or "Geometric" is in its label.
        # If "Power Law" should explicitly not do negative, that's handled if it doesn't meet the Geo/Poisson criteria.

        if process_negative_assortativity:
            print(f"    Processing negative assortativity for {network_label} with std_dev: {std:.2f}")
            net_neg, _, r_neg, c_neg, _ = assortative_network_sample_main(base_network_orig, std, max_attempts_assort, 'negative')
            avg_peak_neg_list = []
            # Assuming net_neg is always successfully created
            available_nodes_neg = list(net_neg.adj.keys()) if net_neg.adj else list(range(net_neg.num_nodes))
            for _ in range(num_avg_runs):
                current_initial_infected_neg = random.sample(available_nodes_neg, k=num_nodes_to_infect_per_trial)
                _, I_counts, _ = simulate_epidemic(net_neg, current_initial_infected_neg, lam_val, num_weeks_sim)
                avg_peak_neg_list.append(max(I_counts) if I_counts else 0)

            r_points.append(r_neg)
            c_points.append(c_neg)
            peak_infection_points.append(np.mean(avg_peak_neg_list))
        else:
            print(f"    Skipping negative assortativity for {network_label} with std_dev: {std:.2f} due to type/std condition.")
            
    return r_points, c_points, peak_infection_points

# --- Main Execution and Plotting ---
if __name__ == "__main__":
    n_nodes_main = 10000
    k_neighbors_main = 20
    power_law_alpha = 2
    ws_p_rewire_values1 = np.linspace(0,0.2,200)
    ws_p_rewire_values2 = np.linspace(0.201,0.6,100)
    ws_p_rewire_values_main = np.concatenate((ws_p_rewire_values1, ws_p_rewire_values2))
    
    ws_plot_color = 'gold'
    ws_legend_name = f"Watts-Strogatz (N={n_nodes_main}, k={k_neighbors_main}, various p)"

    network_configs = [
        {
            "name": f"Poisson (k={k_neighbors_main}, N=10000)",
            "network_creator": lambda: Network.from_poisson(k_neighbors_main, n_nodes_main),
            "marker": 'o',
            "base_color": 'blue',
            "apply_assortativity_modification": True
        },
        {
            "name": f"Geometric (p=1/{(k_neighbors_main+1)}, N=10000)",
            "network_creator": lambda: Network.from_geometric(1/(k_neighbors_main+1), n_nodes_main),
            "marker": 's', 
            "base_color": 'green',
            "apply_assortativity_modification": True
        },
        {
            "name": f"Power Law (α={power_law_alpha}, N=10000)",
            "network_creator": lambda alpha_val=power_law_alpha: Network.from_power_law(alpha=alpha_val, num_nodes=10000, min_deg=1, max_deg=100),
            "marker": 'x', # New marker for Power Law
            "base_color": 'red', # New color for Power Law
            "apply_assortativity_modification": True
        }
    ]

    # Commenting out the loop that adds Watts-Strogatz configurations
    #for p_rewire_val in ws_p_rewire_values_main:
    #     network_configs.append({
    #         "name": ws_legend_name, 
    #         "actual_p_rewire": p_rewire_val,
    #         "network_creator": lambda p_r=p_rewire_val: Network.from_networkx_to_custom(
    #             nx.watts_strogatz_graph(n=n_nodes_main, k=k_neighbors_main, p=p_r)
    #         ),
    #         "marker": '^', 
    #         "base_color": ws_plot_color, 
    #         "apply_assortativity_modification": False 
    #     })
    
    lam_values_main = [0.4]  # Reduced for faster testing, adjust as needed

    for lam_val in lam_values_main:
        print(f"\n--- Processing for lambda = {lam_val} ---")
        fig, axs = plt.subplots(1, 2, figsize=(18, 7)) 
        ws_legend_added_ax0 = False # Separate legend tracking for each subplot
        ws_legend_added_ax1 = False

        for config_idx, config in enumerate(network_configs):
            processing_label = config["name"]
            if "Watts-Strogatz" in config["name"] and "actual_p_rewire" in config:
                processing_label = f"Watts-Strogatz (p={config['actual_p_rewire']:.3f})"
            print(f"\n  Processing: {processing_label}")
            
            base_network = config["network_creator"]()
            # initial_infected_nodes sampling is now done per trial inside helper or the else block below

            r_plot_data, c_plot_data, peak_plot_data = [], [], []

            if config.get("apply_assortativity_modification", True): 
                r_plot_data, c_plot_data, peak_plot_data = get_network_peak_infection_data_modified(
                    base_network,
                    std_values_conf, 
                    # initial_infected_nodes, # Removed
                    lam_val,
                    num_weeks_simulation_config,
                    num_avg_runs_peak_config,
                    processing_label,
                    max_attempts_assort_config,
                    num_initial_to_infect_config # Added: number of nodes to infect per trial
                )
            else: 
                g_nx_for_metrics = convert_to_network(base_network)
                r_val = nx.degree_pearson_correlation_coefficient(g_nx_for_metrics) if g_nx_for_metrics.number_of_edges() > 0 else 0
                c_val = nx.average_clustering(g_nx_for_metrics) if g_nx_for_metrics else 0
                
                r_plot_data.append(r_val)
                c_plot_data.append(c_val)

                avg_peak_list_direct = []
                # Assuming base_network is always successfully created and has nodes
                available_nodes_base = list(base_network.adj.keys()) if base_network.adj else list(range(base_network.num_nodes))
                for _ in range(num_avg_runs_peak_config):
                    current_initial_infected_base = random.sample(available_nodes_base, k=num_initial_to_infect_config)
                    _, I_counts, _ = simulate_epidemic(base_network, current_initial_infected_base, lam_val, num_weeks_simulation_config)
                    avg_peak_list_direct.append(max(I_counts) if I_counts else 0)
                peak_plot_data.append(np.mean(avg_peak_list_direct))
            
            # Plotting logic
            current_legend_label = config["name"]
            
            # Subplot 0: Peak vs Assortativity
            plot_label_ax0 = current_legend_label
            if "Watts-Strogatz" in config["name"]:
                if not ws_legend_added_ax0:
                    ws_legend_added_ax0 = True
                else:
                    plot_label_ax0 = None 
            
            if r_plot_data and peak_plot_data: 
                axs[0].scatter(r_plot_data, peak_plot_data, color=config["base_color"], marker=config["marker"], label=plot_label_ax0, alpha=0.7)

            # Subplot 1: Peak vs Clustering
            plot_label_ax1 = current_legend_label
            if "Watts-Strogatz" in config["name"]:
                if not ws_legend_added_ax1: # Check its own flag
                    ws_legend_added_ax1 = True
                else:
                    plot_label_ax1 = None
            
            if c_plot_data and peak_plot_data:
                axs[1].scatter(c_plot_data, peak_plot_data, color=config["base_color"], marker=config["marker"], label=plot_label_ax1, alpha=0.7)

        axs[0].set_xlabel('Assortativity Coefficient (r)')
        axs[0].set_ylabel('Average Peak Infection Size')
        axs[0].set_title(f'Peak Infection vs. Assortativity')
        axs[0].legend(fontsize='small')
        axs[0].grid(True)

        axs[1].set_xlabel('Clustering Coefficient (c)')
        axs[1].set_ylabel('Average Peak Infection Size') 
        axs[1].set_title(f'Peak Infection vs. Clustering')
        axs[1].legend(fontsize='small')
        axs[1].grid(True)
        
        infection_word = "infection" if num_initial_to_infect_config == 1 else "infections"
        fig.suptitle(f'Peak Infection Analysis (λ={lam_val:.2f}) for {num_initial_to_infect_config} initial {infection_word}', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
        
        save_filename = f'NEWpeak_infection_metrics_lam_{lam_val:.2f}1infection.png'
        plt.savefig(save_filename)
        print(f"    Plot saved as {save_filename}")

