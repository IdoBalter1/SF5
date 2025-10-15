import numpy as np
import random
from itertools import combinations
from collections import defaultdict
from matplotlib import pyplot as plt
import math
from scipy.cluster.hierarchy import DisjointSet
from scipy.stats import binom
import networkx as nx
# Ensure Network class (with from_networkx_to_custom) and convert_to_network are correctly imported from Main
from Main import Network, naive_create_network, gnp_two_stage, dfs, configuration_model, get_degree_distribution, assortative_negative_network,assortative_network,assortative_network_sample,assortative_network_sample_main,convert_to_network,fast_disjoint_set, simulate_epidemic
plt.style.use('ggplot')  # Optional: Use ggplot style for better aesthetics
print("savedpowasdfsdfer")
print("plt.style.use('ggplot')")
print("Task 3.5: Infection Period Analysis 1 infection")
std_values_low_conf = np.linspace(4.5,20,20)
std_values_low_mid_conf = np.linspace(21,60,40)
std_mid_conf = np.linspace(65,100,30)
std_high_conf = np.linspace(105,300,80)
std_high_conf1 = np.linspace(315,6000,200)
std_high_high_conf = np.linspace(6100,15000,100)
std_values  = np.concatenate((std_values_low_conf,std_values_low_mid_conf, std_mid_conf,std_high_conf,std_high_conf1,std_high_high_conf))
print(len(std_values))
max_attempts_assort_config = 1000
num_initial_to_infect_config = 10
num_weeks_simulation_config = 170 
num_avg_runs_infection_length = 100 


# --- Helper Functions (from your existing script) ---
def peak_infection_metrics(network, initial_infected, lam, num_weeks):
    _, I_counts, _ = simulate_epidemic(network, initial_infected, lam, num_weeks)
    peak_size = max(I_counts) if I_counts else 0
    time_to_peak = I_counts.index(peak_size) if peak_size > 0 else 0
    return peak_size, time_to_peak

def length_of_infection_period(network, initial_infected, lam, num_weeks):
    S_counts, I_counts, R_counts = simulate_epidemic(network, initial_infected, lam, num_weeks)
    if not I_counts: return 0 # Handle empty I_counts
    for week in range(1, len(I_counts)):
        if I_counts[week] == 0 and I_counts[week-1] > 0 : # Ensure it was infected before
             return week 
        if I_counts[week] == 0 and week == 1 and I_counts[0] == 0: # No infection at all
             return 0
    return num_weeks

# --- New Data Generation Function ---
def get_network_infection_data(base_network_orig, std_dev_values, 
                               # initial_infected_nodes, # Removed
                               lam_val, num_weeks_sim, num_avg_runs, 
                               network_label_for_condition, max_attempts_assort,
                               num_nodes_to_infect_per_trial # Added
                               ):
    r_points = []
    c_points = []
    time_points = []

    # Data for modified networks
    for std in std_dev_values:
        print(f"    Processing {network_label_for_condition} with std_dev: {std:.2f} for assortativity modification")
        if "Geometric" in network_label_for_condition and std <= 10:
            print(f"      Skipping std={std:.2f} for Geometric network (std <= 10)")
            # Optionally append placeholders if you need consistent list lengths
            # r_points.append(np.nan) 
            # c_points.append(np.nan)
            # time_points.append(np.nan)
            continue

        # Positive assortativity
        process_negative_assortativity = False
        if ("Geometric" in network_label_for_condition and std > 100) or \
           ("Poisson" in network_label_for_condition):
            process_negative_assortativity = True


   
        net_pos, _, r_pos, c_pos, _ = assortative_network_sample_main(base_network_orig, std, max_attempts_assort, 'positive')
        avg_time_pos_list = []
        # Assuming net_pos is always successfully created and has nodes
        available_nodes_pos = list(net_pos.adj.keys()) if net_pos.adj else list(range(net_pos.num_nodes))
        for _ in range(num_avg_runs):
            current_initial_infected_pos = random.sample(available_nodes_pos, k=num_nodes_to_infect_per_trial)
            time = length_of_infection_period(net_pos, current_initial_infected_pos, lam_val, num_weeks_sim)
            avg_time_pos_list.append(time)
        r_points.append(r_pos)
        c_points.append(c_pos)
        time_points.append(np.mean(avg_time_pos_list))


        if process_negative_assortativity:
            net_neg, _, r_neg, c_neg, _ = assortative_network_sample_main(base_network_orig, std, max_attempts_assort, 'negative')
            avg_time_neg_list = []
            # Assuming net_neg is always successfully created and has nodes
            available_nodes_neg = list(net_neg.adj.keys()) if net_neg.adj else list(range(net_neg.num_nodes))
            for _ in range(num_avg_runs):
                current_initial_infected_neg = random.sample(available_nodes_neg, k=num_nodes_to_infect_per_trial)
                time = length_of_infection_period(net_neg, current_initial_infected_neg, lam_val, num_weeks_sim)
                avg_time_neg_list.append(time)
            r_points.append(r_neg)
            c_points.append(c_neg)
            time_points.append(np.mean(avg_time_neg_list))
            
    return r_points, c_points, time_points

# --- Main Execution and Plotting ---
if __name__ == "__main__":
    ws_n_nodes = 10000
    ws_k_neighbors = 20
    watts_strogatz_p_rewire_values1 = np.linspace(0,0.1,100)
    watts_strogatz_p_rewire_values2 = np.linspace(0.102,0.6,200)
    watts_strogatz_p_rewire_values = np.concatenate((watts_strogatz_p_rewire_values1, watts_strogatz_p_rewire_values2))
    # Define a single color and name for all Watts-Strogatz plots
    watts_strogatz_plot_color = 'gold'
    watts_strogatz_legend_name = f"Watts-Strogatz (N={ws_n_nodes}, k={ws_k_neighbors}, various p)"

    # Define alpha for Power Law network
    power_law_alpha = 2

    base_network_configs = [ # Renamed to avoid confusion inside the lambda loop
         {
             "name": "Poisson (k=20, N=10000)",
             "network_creator": lambda: Network.from_poisson(20, 10000),
             "marker": 'o',
             "base_color": 'blue',
             "apply_assortativity_modification": True
         },
         {
             "name": "Geometric (p=1/21, N=10000)",
             "network_creator": lambda: Network.from_geometric(1/21, 10000),
             "marker": 's', # Changed marker for better distinction
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
    
    lambda_values_to_process = [1]
    for current_lam_val in lambda_values_to_process:
        print(f"\n--- Processing for Lambda = {current_lam_val:.2f} ---")
        
        network_configs = [] # Reset for each lambda, especially if WS configs are added dynamically
        for base_config in base_network_configs: # Copy base configs
            network_configs.append(base_config.copy())

        #Dynamically add configurations for each Watts-Strogatz p_rewire value
        # Comment out this loop if you only want Power Law
        for p_rewire_val in watts_strogatz_p_rewire_values:
             network_configs.append({
                 "name": watts_strogatz_legend_name, 
                 "actual_p_rewire": p_rewire_val,
                 "network_creator": lambda p_r=p_rewire_val: Network.from_networkx_to_custom(
                     nx.watts_strogatz_graph(n=ws_n_nodes, k=ws_k_neighbors, p=p_r)
                 ),
                 "marker": 'o', # Or specific marker for WS if desired, e.g., '^'
                 "base_color": watts_strogatz_plot_color, 
                 "apply_assortativity_modification": False 
             })
        
        fig, axs = plt.subplots(1, 2, figsize=(18, 7)) 
        ws_legend_added = False # Reset for each new plot/lambda value

        for config in network_configs:
            # Adjust print statement to show specific p_rewire for WS if desired
            if "Watts-Strogatz" in config["name"] and "actual_p_rewire" in config:
                print(f"\n  Processing base network type: Watts-Strogatz (p={config['actual_p_rewire']:.3f}) for lambda={current_lam_val:.2f}")
            else:
                print(f"\n  Processing base network type: {config['name']} for lambda={current_lam_val:.2f}")
            
            base_network = config["network_creator"]() 
            
            r_data, c_data, time_data = [], [], []

            processing_label = config["name"]
            if "Watts-Strogatz" in config["name"] and "actual_p_rewire" in config:
                 processing_label = f"Watts-Strogatz (p={config['actual_p_rewire']:.3f})"


            if config.get("apply_assortativity_modification", True): 
                r_data, c_data, time_data = get_network_infection_data(
                    base_network,
                    std_values,
                    current_lam_val, # Use current lambda
                    num_weeks_simulation_config,
                    num_avg_runs_infection_length,
                    processing_label, 
                    max_attempts_assort_config,
                    num_initial_to_infect_config
                )
            else: 
                print(f"    Processing {processing_label} directly (no assortativity modification loop).")
                g_nx_for_metrics = convert_to_network(base_network) 
                
                r_val = nx.degree_pearson_correlation_coefficient(g_nx_for_metrics) if g_nx_for_metrics.number_of_edges() > 0 else 0.0
                c_val = nx.average_clustering(g_nx_for_metrics) if g_nx_for_metrics else 0.0
                
                r_data.append(r_val)
                c_data.append(c_val)

                avg_time_list = []
                available_nodes_base = list(base_network.adj.keys()) if base_network.adj else list(range(base_network.num_nodes))
                for _ in range(num_avg_runs_infection_length):
                    current_initial_infected_base = random.sample(available_nodes_base, k=num_initial_to_infect_config)
                    time = length_of_infection_period(base_network, current_initial_infected_base, current_lam_val, num_weeks_simulation_config) # Use current lambda
                    avg_time_list.append(time)
                time_data.append(np.mean(avg_time_list))
            
            current_label_for_plot = config["name"] # Use a different variable for plot label
            if "Watts-Strogatz" in config["name"]:
                if not ws_legend_added:
                    plot_label_to_use = current_label_for_plot
                    ws_legend_added = True
                else:
                    plot_label_to_use = None 
            else:
                plot_label_to_use = current_label_for_plot


            if r_data: 
                axs[0].scatter(r_data, time_data, color=config["base_color"], marker=config["marker"], label=plot_label_to_use, alpha=0.7)
            if c_data:
                axs[1].scatter(c_data, time_data, color=config["base_color"], marker=config["marker"], label=plot_label_to_use, alpha=0.7)

        axs[0].set_xlabel('Assortativity Coefficient (r)')
        axs[0].set_ylabel('Average Length of Infection Period (Weeks)')
        axs[0].set_title(f'Infection Length vs. Assortativity (λ={current_lam_val:.2f})') # Use current lambda
        axs[0].legend(fontsize='small')
        axs[0].grid(True)

        axs[1].set_xlabel('Clustering Coefficient (C_g)')
        axs[1].set_ylabel('Average Length of Infection Period (Weeks)') 
        axs[1].set_title(f'Infection Length vs. Clustering (λ={current_lam_val:.2f})') # Use current lambda
        axs[1].legend(fontsize='small')
        axs[1].grid(True)

        infection_word = "infection" if num_initial_to_infect_config == 1 else "infections"
        fig.suptitle(f'Infection Period Analysis (λ={current_lam_val:.2f}) for {num_initial_to_infect_config} initial {infection_word}', fontsize=16)
        plt.tight_layout()
        filename = f"NEWNInfection_period_analysis_lambda_{current_lam_val:.2f}_1infection0_2.svg" # Use current lambda in filename
        plt.savefig(filename)
        print(f"Plot saved to {filename}")
        plt.close(fig) # Close the figure to free memory before the next lambda iteration

    print("\nAll processing complete.")