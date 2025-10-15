from Main import Network, naive_create_network,assortative_network_sample_main,convert_to_network, gnp_two_stage, dfs, configuration_model, get_degree_distribution,simulate_epidemic,estimate_outbreak_disjointset_from_network
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from collections import defaultdict
from itertools import combinations
import math
import timeit
import networkx as nx
print("saved")
print("11111Task 3.4: Assortative Nesdfsdfsdftwork Samplsdfsdfe Main")
network = Network.from_power_law(2, 10000,1,100)
G1 = convert_to_network(network)

r = nx.degree_pearson_correlation_coefficient(G1)
c = nx.average_clustering(G1)
p_critical = 0.05
lambda_values = np.linspace(0, 1, 300)
# Corrected typo: np.linspace instead of np.linsapce
std_values = [5,10,15,20,30]
print("2sk 3.4: Plottig for multiple std_dsdfdev values") # Updated print

# Create figure and subplots once
fig, axs = plt.subplots(1, 2, figsize=(12, 5)) # axs[0] for mean, axs[1] for std/mean
r_vals = []
current_mean_sizes1 = []
current_std_over_mean_sizes1 = []
clustering_coefficients = []
r_vals.append(r)
clustering_coefficients.append(c)

print("2)")
for lam in lambda_values:
    print(f"  Lambda: {lam:.3f}", end='')
    all_sizes_for_lam1 = [] # Renamed for clarity
    for _ in range(2):
         # 3 runs for averaging at this lambda
        # NOTE: std_dev_val is NOT used in this function call in your original code.
        # If std_dev_val is supposed to modify the network or the estimation process,
        # estimate_outbreak_disjointset_from_network would need to accept it as an argument,
        # or the network itself would need to be regenerated using std_dev_val.
        sizes1 = estimate_outbreak_disjointset_from_network(network, lam)
        all_sizes_for_lam1.extend(sizes1)
        
    mean1 = np.mean(all_sizes_for_lam1)
    std1 = np.std(all_sizes_for_lam1)
    # print(f"  Lambda: {lam:.3f}, Mean Size: {mean:.2f}, Std: {std:.2f}") # Indent for clarity
    
    current_mean_sizes1.append(mean1)
    current_std_over_mean_sizes1.append(std1 / mean1 if mean1 != 0 else 0)
for std_dev_val in std_values: # Use a more descriptive variable name
    new_net,_, r,c,G = assortative_network_sample_main(network, std_dev_val, 1000, 'negative')
    clustering_coefficients.append(c)
    print(f"  Assortativity Coefficient: {r:.2f}, Clustering Coefficient: {c:.2f}") # Show progress
    current_mean_sizes = []
    current_std_over_mean_sizes = []
    r_vals.append(r)

    
    print(f"\nProcessing for std_dev_val: {std_dev_val:.2f}") # Show progress

    for lam in lambda_values:
        print(f"  Lambda: {lam:.3f}", end='')
        all_sizes_for_lam = [] # Renamed for clarity
        for _ in range(2): 
            # NOTE: std_dev_val is NOT used in this function call in your original code.
            # If std_dev_val is supposed to modify the network or the estimation process,
            # estimate_outbreak_disjointset_from_network would need to accept it as an argument,
            # or the network itself would need to be regenerated using std_dev_val.
            sizes = estimate_outbreak_disjointset_from_network(new_net, lam)
            all_sizes_for_lam.extend(sizes)
            
        mean = np.mean(all_sizes_for_lam)
        std = np.std(all_sizes_for_lam)
        # print(f"  Lambda: {lam:.3f}, Mean Size: {mean:.2f}, Std: {std:.2f}") # Indent for clarity
        
        current_mean_sizes.append(mean)
        current_std_over_mean_sizes.append(std / mean if mean != 0 else 0)

    # Plot results for the current std_dev_val on the subplots
    axs[0].plot(lambda_values, current_mean_sizes, 'o-', markersize=3, label=f'C_g= {c:.4f}, r = {r:.4f}')
    axs[1].plot(lambda_values, current_std_over_mean_sizes, 'o-', markersize=3, label=f'C_g= {c:.4f}, r ={r:.4f}')

# Configure subplot 1 (Mean Size vs Lambda)
axs[0].set_xlabel('Lambda')
axs[0].set_ylabel('Mean Size')
axs[0].set_title('Mean Size vs Lambda')
axs[0].legend(loc='upper left', fontsize='small', ncol = 1) # Adjusted legend
axs[0].grid(True)

# Configure subplot 2 (Std/Mean Size vs Lambda)
#axs[1].axvline(x = p_critical, color='k', linestyle='--', label=f'Critical λ ({p_critical:.2f})') # Changed color for visibility
axs[1].set_xlabel('Lambda')
axs[1].set_ylabel('Std/Mean Size')
axs[1].set_title('Std/Mean Size vs Lambda')
axs[1].legend(loc='upper right', fontsize='small', ncol = 1) # Adjusted legend
axs[1].grid(True)

plt.tight_layout()
figname = 'std_over_mean_poisson.png'  # Define the filename
plt.savefig(figname)