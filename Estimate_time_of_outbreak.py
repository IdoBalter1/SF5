import numpy as np
import random
from itertools import combinations
from collections import defaultdict
from matplotlib import pyplot as plt
import math
from scipy.cluster.hierarchy import DisjointSet
from scipy.stats import binom
import networkx as nx
from Main import Network, naive_create_network, gnp_two_stage, dfs, configuration_model, get_degree_distribution, assortative_negative_network,assortative_network,assortative_network_sample,assortative_network_sample_main,convert_to_network,fast_disjoint_set, simulate_epidemic
std_values_low = np.linspace(4.5,20,20)
std_values_low_mid = np.linspace(21,60,20)
std_mid = np.linspace(65,100,10)
std_high = np.linspace(105,300,20)
std_high_high = np.linspace(350,8000,20) # Reduced for faster example, adjust as needed
std_values  = np.concatenate((std_values_low,std_values_low_mid, std_mid,std_high,std_high_high))

print("std_values",std_values)
max_attempts = 1000
value = 'positive'
lambda_vals = np.linspace(0, 1, 100)
num_initial_to_infect_config = 1
num_weeks_simulation_config = 70
max_attempts_assort_config = 1000
assortativity_type_config = 'positive' # or 'negative'
num_avg_runs_infection_length = 5 # Number of simulation runs to average for infection length
def peak_infection_metrics(network, initial_infected, lam, num_weeks):
    _, I_counts, _ = simulate_epidemic(network, initial_infected, lam, num_weeks)
    peak_size = max(I_counts)
    time_to_peak = I_counts.index(peak_size)
    return peak_size, time_to_peak

def length_of_infection_period(network, initial_infected, lam, num_weeks):
    """
    Simulate the epidemic and return the length of the infection period.
    The infection period is defined as the number of weeks until no new infections occur.
    """
    S_counts, I_counts, R_counts = simulate_epidemic(network, initial_infected, lam, num_weeks)
    
    # Find the first week where no new infections occurred
    for week in range(1, len(I_counts)):
        if I_counts[week] == 0:
            return week  # Return the week when no infections occurred
    
    return num_weeks  # If all weeks had infections, return num_weeks

#make a plot of length of time against lambda values for different networks
def length_for_different_networks(network,initial_infected, lam, num_weeks):
    """
    Calculate the length of the infection period for different networks.
    
    """
    value1 = 'positive'
    plt.figure(figsize=(10, 6))
    r_values = []
    time_results = []
    cluster_values = []
    g1 =  convert_to_network(network)
    r1 = nx.degree_pearson_correlation_coefficient(g1)
    r_values.append(r1)
    c1 = nx.average_clustering(g1)
    cluster_values.append(c1)
    time_results1 = []
    for i in range(1000):
        time1 = length_of_infection_period(network, initial_infected, lam, num_weeks)
        time_results1.append(time1)
    time_results.append(np.mean(time_results1))
    for std_dev in std_values:
        print(f"Sampling with standard deviation: {std_dev}")
        new_net, degrees, r,c,G = assortative_network_sample_main(network,std_dev,max_attempts,'positive')
        new_net2, degrees2, r2,c2,G2 = assortative_network_sample_main(network,std_dev,max_attempts,'negative')
        r_values.append(r)
        r_values.append(r2)
        cluster_values.append(c)
        cluster_values.append(c2)
        print(f"clustering coefficient for positive assortativity: {c2}")
        average_time = []
        average_time2 = []
        for i in range(1000):
            time = length_of_infection_period(new_net, initial_infected, lam, num_weeks)
            time2 = length_of_infection_period(new_net2, initial_infected, lam, num_weeks)
            average_time2.append(time2)
            average_time.append(time)

       
        time_results.append(np.mean(average_time))
        time_results.append(np.mean(average_time2))
        
    plt.scatter(r_values, time_results, label=f'Infection Length (λ={lam:.2f})', marker='o', alpha=0.7)
    
    plt.xlabel('Assortativity Coefficient (r)') # Corrected X-axis label
    plt.ylabel('Average Length of Infection Period (Weeks)')
    # Assuming initial_infected_nodes_for_base was selected by highest degree
    title_detail = f"Initial Infected: Top {len(initial_infected)} Highest Degree Nodes (from base network)"
    plt.title(f'Infection Length vs. Assortativity (λ={lam:.2f}, for one initially infected node)')
    plt.legend()
    plt.grid(True)
    figname = f'infection_length_vs_assortativity_lambda_{lam:.2f}.png'
    plt.savefig(figname)
    return new_net

network = Network.from_poisson(20, 10000)

initial_infected = random.sample(list(network.adj.keys()), k=1)
lambda_vals = [0.2]
for lam in lambda_vals:
    x = length_for_different_networks(network,initial_infected,lam,170)        

