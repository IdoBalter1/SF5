import numpy as np
import random
from itertools import combinations
from collections import defaultdict
from matplotlib import pyplot as plt
import math
from scipy.cluster.hierarchy import DisjointSet
from scipy.stats import binom
import networkx as nx
from Main import Network, naive_create_network, gnp_two_stage, dfs, configuration_model, get_degree_distribution, assortative_negative_network,assortative_network,assortative_network_sample,assortative_network_sample_main,convert_to_network,fast_disjoint_set
print("savedpowerlawsdfdfsampledifferent xsdf")
plt.style.use('ggplot')
print("plt.style.use('ggplot')")
# These global variables are used by the function
std_values_low_conf = np.linspace(4.5,20,40)
std_values_low_mid_conf = np.linspace(21,60,40)
std_mid_conf = np.linspace(65,100,40)
std_high_conf = np.linspace(105,300,80)
std_high_high_conf = np.linspace(350,1500,150)
std_values  = np.concatenate((std_values_low_conf,std_values_low_mid_conf, std_mid_conf,std_high_conf,std_high_high_conf))
print(len(std_values))
max_attempts_assort_mod = 1000 # Renamed to avoid conflict if 'max_attempts' is used elsewhere with different meaning
num_avg_runs_fds = 100 # Specific to averaging FDS runs
network_modification_type = 'positive' # Or 'negative', or you could loop through types

# network_template: The base network to modify.
# all_lambdas_for_fds_calculation: The full array of lambdas that fast_disjoint_set uses for its calculations.
# specific_lambdas_to_plot: A list of the specific lambda values for which you want to generate plots.
# x_node_idx_for_fds: The 'x' parameter for fast_disjoint_set (used consistently for all FDS calls here).
def estimate_cluster_sizes(
    network_template,
    all_lambdas_for_fds_calculation,
    specific_lambdas_to_plot # This now expects a list of lambdas
    # x_node_idx_for_fds parameter removed
):
    
    # --- STAGE 1: Data Collection ---
    collected_network_data = [] 

    print("Stage 1: Calculating network properties and outbreak sizes for all lambdas...")
    for std_dev in std_values:
        # print(f"  Processing std_dev: {std_dev:.2f} for {network_modification_type} modification") 
        # Pass network_modification_type to the function
        new_net, degrees, current_r,current_c,G_assortative = assortative_network_sample_main(
            network_template, std_dev, max_attempts_assort_mod, network_modification_type
        )

        # Removed: if new_net is None: check

        all_fds_runs_for_this_new_net = []
        # print(f"    Running fast_disjoint_set {num_avg_runs_fds} times for averaging...")
        for _ in range(num_avg_runs_fds): # Use num_avg_runs_fds
            # Sample a new x_node_idx for each FDS run from the current new_net
            # Assuming new_net.num_nodes is always > 0 and nodes are 0-indexed
            current_x_node_idx = random.randint(0, new_net.num_nodes - 1) 
            cluster_outputs_one_run = fast_disjoint_set(new_net, all_lambdas_for_fds_calculation, current_x_node_idx)
            all_fds_runs_for_this_new_net.append(cluster_outputs_one_run)
        
        # Removed: if all_fds_runs_for_this_new_net: and else block
        np_all_fds_runs = np.array(all_fds_runs_for_this_new_net)
        averaged_cluster_outputs = np.mean(np_all_fds_runs, axis=0).tolist()
        
        collected_network_data.append({
            'r': current_r,
            'c': current_c,
            'all_outbreaks': averaged_cluster_outputs # CORRECTED VARIABLE
        })
        
    print(f"Stage 1 complete. Collected data for {len(collected_network_data)} network variations (based on std_dev).")

    # --- STAGE 2: Plotting for each specific lambda in the provided list ---
    # Removed: if not collected_network_data: check

    print(f"\nStage 2: Generating {len(specific_lambdas_to_plot)} plots...")
    np_all_lambdas_for_fds_calculation = np.array(all_lambdas_for_fds_calculation)

    for specific_lam_for_current_plot in specific_lambdas_to_plot:
        r_values_for_this_plot = []
        c_values_for_this_plot = []
        outbreak_sizes_for_this_plot = []

        idx_for_specific_lam = np.abs(np_all_lambdas_for_fds_calculation - specific_lam_for_current_plot).argmin()
        actual_lambda_value_from_array = np_all_lambdas_for_fds_calculation[idx_for_specific_lam]

        for network_data_item in collected_network_data:
            r_values_for_this_plot.append(network_data_item['r'])
            c_values_for_this_plot.append(network_data_item['c'])
            outbreak_size_for_current_r = network_data_item['all_outbreaks'][idx_for_specific_lam]
            outbreak_sizes_for_this_plot.append(outbreak_size_for_current_r)
        
        if not r_values_for_this_plot: # Check if data exists for this lambda
            print(f"  No data to plot for lambda approx {actual_lambda_value_from_array:.2f}. Skipping this plot.")
            continue

        fig, axs = plt.subplots(1, 2, figsize=(15, 6))

        axs[0].scatter(r_values_for_this_plot, outbreak_sizes_for_this_plot, label=f'λ≈{actual_lambda_value_from_array:.2f}', marker='o', alpha=0.7)
        axs[0].set_xlabel('Assortativity Coefficient (r)')
        axs[0].set_ylabel('Average Outbreak Size') # Changed for clarity
        axs[0].set_title(f'vs. Assortativity (r)')
        axs[0].legend()
        axs[0].grid(True)

        axs[1].scatter(c_values_for_this_plot, outbreak_sizes_for_this_plot, label=f'λ≈{actual_lambda_value_from_array:.2f}', marker='x', alpha=0.7)
        axs[1].set_xlabel('Clustering Coefficient (C_g)')
        axs[1].set_ylabel('Average Outbreak Size') # Changed for clarity
        axs[1].set_title(f'vs. Clustering (C_g)')
        axs[1].legend()
        axs[1].grid(True)

        fig.suptitle(f'Outbreak Size Analysis ({network_modification_type} assortativity) for λ ≈ {actual_lambda_value_from_array:.2f}', fontsize=16)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Improved filename and added extension
        save_filename = f'estimate_cluster_size_{network_modification_type}_lambda_{actual_lambda_value_from_array:.3f}1.png'
        plt.savefig(save_filename)
        print(f"  Plot saved: {save_filename}1")
        # plt.show() # Uncomment to display plots during execution
        plt.close(fig) # Close the figure to free memory, especially important in loops
    
    print("Stage 2 complete. All requested plots generated.")

# --- Corrected Calling Code ---
if __name__ == "__main__":
    # p_val_for_template = 1/(21) # Example, if your template used it
    network_template_main = Network.from_power_law(2,10000,1,100) # Your base network
    
    num_nodes_main = network_template_main.num_nodes

    all_lambdas_for_calculation = np.arange(0, 1.01, 0.01).tolist()

    specific_lambdas_for_plotting = [0.02,0.05,0.1,0.15,0.2,0.4,1.0]
    
    # You could extend this to loop through modification types:
    # for mod_type in ['positive', 'negative']:
    #     global network_modification_type # If you want to change the global for each run
    #     network_modification_type = mod_type
    #     print(f"\nRunning analysis for {mod_type} assortativity modification.")
    #     estimate_cluster_sizes(
    #         network_template_main,
    #         all_lambdas_for_calculation,
    #         specific_lambdas_for_plotting,
    #         x_node_idx_main
    #     )

    print(f"Calling estimate_cluster_sizes for {network_modification_type} modification to generate {len(specific_lambdas_for_plotting)} plots.")
    estimate_cluster_sizes(
        network_template_main,
        all_lambdas_for_calculation,
        specific_lambdas_for_plotting
    )
    print("Script finished.")