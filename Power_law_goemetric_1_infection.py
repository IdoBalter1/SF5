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

print("Task 3.5: Peak Infection Metrics (Power Law & Geometric)") 
plt.style.use('ggplot') 
print("plt.style.use('ggplot')") 
print("plt.style.use('ggplot')")  # Optional: Use ggplot style for better aesthetics

# --- Configuration for Assortativity Modification ---
std_values_low_conf = np.linspace(4.5,20,20)
std_values_low_mid_conf = np.linspace(21,60,40)
std_mid_conf = np.linspace(65,100,30)
std_high_conf = np.linspace(105,300,30)
std_high_high_conf = np.linspace(350,5500,100)
std_values_conf  = np.concatenate((std_values_low_conf,std_values_low_mid_conf, std_mid_conf,std_high_conf,std_high_high_conf))
print(f"Number of STD values for assortativity modification: {len(std_values_conf)}")

max_attempts_assort_config = 1000
num_initial_to_infect_config = 1
num_weeks_simulation_config = 150
num_avg_runs_peak_config = 100 

# --- Helper function for networks undergoing assortativity modification ---
def get_network_peak_infection_data_modified(
    base_network_orig,
    std_dev_values_list,
    lam_val,
    num_weeks_sim,
    num_avg_runs,
    network_label, 
    max_attempts_assort,
    num_nodes_to_infect_per_trial
):
    r_points = []
    c_points = []
    peak_infection_points = []

    for std in std_dev_values_list:
        print(f"    Processing {network_label} with std_dev: {std:.2f} for assortativity modification")
            
        # Positive assortativity
        # Corrected condition for case sensitivity and clarity with parentheses
        if ("Power Law" in network_label) or ("Geometric" in network_label and std > 7.5):
            net_pos, _, r_pos, c_pos, _ = assortative_network_sample_main(base_network_orig, std, max_attempts_assort, 'positive')
            avg_peak_pos_list = []
            # Assuming net_pos.adj or net_pos.num_nodes provides sufficient nodes
            available_nodes_pos = list(net_pos.adj.keys()) if net_pos.adj else list(range(net_pos.num_nodes))
            # Assuming available_nodes_pos will always have enough nodes for sampling
            for _ in range(num_avg_runs):
                current_initial_infected_pos = random.sample(available_nodes_pos, k=num_nodes_to_infect_per_trial)
                _, I_counts, _ = simulate_epidemic(net_pos, current_initial_infected_pos, lam_val, num_weeks_sim)
                avg_peak_pos_list.append(max(I_counts)) # Assuming I_counts is never empty
            
            r_points.append(r_pos)
            c_points.append(c_pos)
            peak_infection_points.append(np.mean(avg_peak_pos_list)) # Assuming avg_peak_pos_list is never empty

        # Negative assortativity (currently commented out)
        if "Geometric" in network_label and std>10:
            net_neg, _, r_neg, c_neg, _ = assortative_network_sample_main(base_network_orig, std, max_attempts_assort, 'negative')
            avg_peak_neg_list = []
            available_nodes_neg = list(net_neg.adj.keys()) if net_neg.adj else list(range(net_neg.num_nodes))
            for _ in range(num_avg_runs):
                current_initial_infected_neg = random.sample(available_nodes_neg, k=num_nodes_to_infect_per_trial)
                _, I_counts, _ = simulate_epidemic(net_neg, current_initial_infected_neg, lam_val, num_weeks_sim)
                avg_peak_neg_list.append(max(I_counts)) 

            r_points.append(r_neg)
            c_points.append(c_neg)
            peak_infection_points.append(np.mean(avg_peak_neg_list))
            
    return r_points, c_points, peak_infection_points

# --- Main Execution and Plotting ---
if __name__ == "__main__":
    n_nodes_main = 10000 
    power_law_alpha = 2
    geometric_p = 1/21
    
    network_configs = [
        {
            "name": f"Power Law (α={power_law_alpha}, N={n_nodes_main})",
            "network_creator": lambda alpha_val=power_law_alpha: Network.from_power_law(
                alpha=alpha_val, 
                num_nodes=n_nodes_main, 
                min_deg=1, 
                max_deg=100 
            ),
            "marker": 'x', 
            "base_color": 'red', 
            "apply_assortativity_modification": True
        },
        {
            "name": f"Geometric (p={geometric_p:.3f}, N={n_nodes_main})",
            "network_creator": lambda p_val=geometric_p: Network.from_geometric(
                p=p_val,
                num_nodes=n_nodes_main
            ),
            "marker": 's', 
            "base_color": 'green', 
            "apply_assortativity_modification": True 
        }
    ]
    
    lam_values_main = [0.1,0.4,1.0] 

    for lam_val in lam_values_main:
        print(f"\n--- Processing Networks for lambda = {lam_val} ---")
        fig, axs = plt.subplots(1, 2, figsize=(18, 7)) 

        for config in network_configs: 
            processing_label = config["name"]
            print(f"\n  Processing: {processing_label}")
            
            base_network = config["network_creator"]()
            
            r_plot_data, c_plot_data, peak_plot_data = [], [], []

            if config["apply_assortativity_modification"]:
                r_plot_data, c_plot_data, peak_plot_data = get_network_peak_infection_data_modified(
                    base_network,
                    std_values_conf, 
                    lam_val,
                    num_weeks_simulation_config,
                    num_avg_runs_peak_config,
                    processing_label, 
                    max_attempts_assort_config,
                    num_initial_to_infect_config
                )
            else: 
                # This block is for networks not undergoing std_dev modification.
                # Assuming base_network is valid and has enough nodes.
                print(f"    Processing {processing_label} directly (no assortativity modification loop).")
                g_nx_for_metrics = convert_to_network(base_network)
                r_val = nx.degree_pearson_correlation_coefficient(g_nx_for_metrics) # Assuming number_of_edges > 0
                c_val = nx.average_clustering(g_nx_for_metrics) # Assuming g_nx_for_metrics is not None
                
                avg_peak_list_base = []
                available_nodes_base = list(base_network.adj.keys()) if base_network.adj else list(range(base_network.num_nodes))
                # Assuming available_nodes_base has enough nodes
                for _ in range(num_avg_runs_peak_config):
                    current_initial_infected_base = random.sample(available_nodes_base, k=num_initial_to_infect_config)
                    _, I_counts, _ = simulate_epidemic(base_network, current_initial_infected_base, lam_val, num_weeks_simulation_config)
                    avg_peak_list_base.append(max(I_counts)) # Assuming I_counts not empty
                peak_val_base = np.mean(avg_peak_list_base) # Assuming avg_peak_list_base not empty
                
                r_plot_data.append(r_val)
                c_plot_data.append(c_val)
                peak_plot_data.append(peak_val_base)

            plot_label = config["name"] 
            
            # Assuming r_plot_data, c_plot_data, peak_plot_data do not contain NaNs
            axs[0].scatter(r_plot_data, peak_plot_data, color=config["base_color"], marker=config["marker"], label=plot_label, alpha=0.7)
            axs[1].scatter(c_plot_data, peak_plot_data, color=config["base_color"], marker=config["marker"], label=plot_label, alpha=0.7)

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
        
        fig.suptitle(f'Peak Infection Analysis (λ={lam_val:.2f}, {num_initial_to_infect_config} initial infections)', fontsize=16) 
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
        
        save_filename = f'peak_infection_PL_Geo_lam_{lam_val:.2f}_{num_initial_to_infect_config}infections.png' 
        plt.savefig(save_filename)
        print(f"    Plot saved as {save_filename}")
        plt.close(fig) 
    print("\nProcessing complete for all lambda values.")