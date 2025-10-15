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

print("Task 3.5: Peak Infection Metrics (Power Law Only)")
plt.style.use('ggplot') 
print("plt.style.use('ggplot')") # Optional: Use ggplot style for better aesthetics

# --- Configuration for Assortativity Modification ---
std_values_low_conf = np.linspace(4.5,20,10)
std_values_low_mid_conf = np.linspace(21,60,10)
std_mid_conf = np.linspace(65,100,10)
std_high_conf = np.linspace(105,300,80)
std_high_high_conf = np.linspace(350,10000,250)
std_values_conf  = np.concatenate((std_values_low_conf,std_values_low_mid_conf, std_mid_conf,std_high_conf,std_high_high_conf))
print(f"Number of STD values for assortativity modification: {len(std_values_conf)}")

max_attempts_assort_config = 1000
num_initial_to_infect_config = 10
num_weeks_simulation_config = 150
num_avg_runs_peak_config = 1 # Number of simulation runs to average peak size

# --- Helper function for networks undergoing assortativity modification ---
def get_network_peak_infection_data_modified(
    base_network_orig,
    std_dev_values_list,
    # initial_infected_nodes, # Removed this parameter
    lam_val,
    num_weeks_sim,
    num_avg_runs,
    network_label, 
    max_attempts_assort,
    num_nodes_to_infect_per_trial # Added: number of nodes to infect for each trial
):
    r_points = []
    c_points = []
    peak_infection_points = []

    for std in std_dev_values_list:
        print(f"    Processing {network_label} with std_dev: {std:.2f} for assortativity modification")
            
        # Positive assortativity
        net_pos, _, r_pos, c_pos, _ = assortative_network_sample_main(base_network_orig, std, max_attempts_assort, 'positive')
        avg_peak_pos_list = []
        # Assuming net_pos is always successfully created and has nodes
        available_nodes_pos = list(net_pos.adj.keys()) if net_pos.adj else list(range(net_pos.num_nodes))
        for _ in range(num_avg_runs):
            current_initial_infected_pos = random.sample(available_nodes_pos, k=num_nodes_to_infect_per_trial)
            _, I_counts, _ = simulate_epidemic(net_pos, current_initial_infected_pos, lam_val, num_weeks_sim)
            avg_peak_pos_list.append(max(I_counts)) 
        
        r_points.append(r_pos)
        c_points.append(c_pos)
        peak_infection_points.append(np.mean(avg_peak_pos_list)) # Assuming avg_peak_pos_list is never empty

        # Negative assortativity
        """net_neg, _, r_neg, c_neg, _ = assortative_network_sample_main(base_network_orig, std, max_attempts_assort, 'negative')
        avg_peak_neg_list = []
        # Assuming net_neg is always successfully created and has nodes
        available_nodes_neg = list(net_neg.adj.keys()) if net_neg.adj else list(range(net_neg.num_nodes))
        for _ in range(num_avg_runs):
            current_initial_infected_neg = random.sample(available_nodes_neg, k=num_nodes_to_infect_per_trial)
            _, I_counts, _ = simulate_epidemic(net_neg, current_initial_infected_neg, lam_val, num_weeks_sim)
            avg_peak_neg_list.append(max(I_counts)) # Assuming I_counts is never empty

        r_points.append(r_neg)
        c_points.append(c_neg)
        peak_infection_points.append(np.mean(avg_peak_neg_list))""" # Assuming avg_peak_neg_list is never empty
            
    return r_points, c_points, peak_infection_points

# --- Main Execution and Plotting ---
if __name__ == "__main__":
    n_nodes_main = 10000 # General number of nodes, used by Power Law
    power_law_alpha = 2
    
    # Define the Power Law network configuration
    network_configs = [
        {
            "name": f"Power Law (α={power_law_alpha}, N=10000)",
            "network_creator": lambda alpha_val=power_law_alpha: Network.from_power_law(
                alpha=alpha_val, 
                num_nodes=n_nodes_main, 
                min_deg=1, 
                max_deg=100 # Max degree for Power Law
            ),
            "marker": 'x', 
            "base_color": 'red', 
            "apply_assortativity_modification": True # Power Law network will be modified
        }
    ]
    
    lam_values_main = [0.1,0.4,1.0] 

    for lam_val in lam_values_main:
        print(f"\n--- Processing Power Law Network for lambda = {lam_val} ---")
        fig, axs = plt.subplots(1, 2, figsize=(18, 7)) 

        for config in network_configs: # This loop will effectively run once
            processing_label = config["name"]
            print(f"\n  Processing: {processing_label}")
            
            base_network = config["network_creator"]()
            
            r_plot_data, c_plot_data, peak_plot_data = [], [], []

            # Assuming apply_assortativity_modification is True for Power Law
            r_plot_data, c_plot_data, peak_plot_data = get_network_peak_infection_data_modified(
                base_network,
                std_values_conf, 
                # initial_infected_nodes, # Removed
                lam_val,
                num_weeks_simulation_config,
                num_avg_runs_peak_config,
                processing_label, 
                max_attempts_assort_config,
                num_initial_to_infect_config # Pass the number of nodes to infect per trial
            )
            
            # Plotting logic - assuming data lists are populated
            plot_label = config["name"] 
            
            axs[0].scatter(r_plot_data, peak_plot_data, color=config["base_color"], marker=config["marker"], label=plot_label, alpha=0.7)
            axs[1].scatter(c_plot_data, peak_plot_data, color=config["base_color"], marker=config["marker"], label=plot_label, alpha=0.7)

        axs[0].set_xlabel('Assortativity Coefficient (r)')
        axs[0].set_ylabel('Average Peak Infection Size')
        axs[0].set_title(f'Peak Infection vs. Assortativity (Power Law)')
        axs[0].legend(fontsize='small')
        axs[0].grid(True)

        axs[1].set_xlabel('Clustering Coefficient (c)')
        axs[1].set_ylabel('Average Peak Infection Size') 
        axs[1].set_title(f'Peak Infection vs. Clustering (Power Law)')
        axs[1].legend(fontsize='small')
        axs[1].grid(True)
        
        fig.suptitle(f'Peak Infection Analysis for Power Law Network (λ={lam_val:.2f})', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
        
        save_filename = f'peak_infection_power_law_lam_{lam_val:.2f}10infections.png'
        plt.savefig(save_filename)
        print(f"    Plot saved as {save_filename}")
        # plt.show() # Uncomment to display plot after each lambda
    print("\nProcessing complete for all lambda values.")

