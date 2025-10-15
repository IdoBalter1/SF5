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
from matplotlib.lines import Line2D # Import for custom legend
print("savedsdf")
print("plt.style.use('ggplot')")
plt.style.use('ggplot')
#std_values_low_conf = np.linspace(4.5,20,10)
#std_values_low_mid_conf = np.linspace(21,60,20)
#std_mid_conf = np.linspace(65,100,20)#
#std_high_conf = np.linspace(105,300,20)#
std_values = np.linspace(2000,150000,100)
#std_values  = np.concatenate((std_values_low_conf,std_values_low_mid_conf, std_mid_conf,std_high_conf,std_high_high_conf))
network_p = Network.from_poisson(20, 100000)
#network_g = Network.from_geometric(1/21, 100000) # Fixed typo: netowrk_g -> network_g
#network_power = Network.from_power_law(2,100000,1,500) # Assuming you have a power-law network defined
# Define colors for each network type
color_poisson = 'blue'
color_geometric = 'green'
color_power_law = 'red'  # If you have a power-law network, define its color

plt.figure(figsize=(10, 6))

# To ensure labels are added only once for the legend
plotted_poisson = False
plotted_geometric = False

for std in std_values:
    print(f"Processing std_dev: {std:.2f}") # Optional: for progress tracking
    
    # Poisson networks
    _, _, r_pos_p, c_pos_p, _ = assortative_network_sample_main(network_p, std, 1000, 'positive')
    #_, _, r_neg_p, c_neg_p, _ = assortative_network_sample_main(network_p, std, 1000, 'negative')
    
    # Geometric networks
    """if std >=7.5:
       # _, _, r_pos_g, c_pos_g, _ = assortative_network_sample_main(network_g, std, 1000, 'positive') # Fixed typo
       # _, _, r_neg_g, c_neg_g, _ = assortative_network_sample_main(network_g, std, 1000, 'negative') # Fixed typo
        plt.scatter(r_pos_g, c_pos_g, color=color_geometric, alpha=0.6)
        plt.scatter(r_neg_g, c_neg_g, color=color_geometric, alpha=0.6)"""
    
    #_,_, r_pos_pl, c_pos_pl, _ = assortative_network_sample_main(network_power, std, 1000, 'positive') # Assuming you have a power-law network
    #_,_, r_neg_pl, c_neg_pl, _ = assortative_network_sample_main(network_power, std, 1000, 'negative') # Assuming you have a power-law network
    # Scatter plots for Poisson
    plt.scatter(r_pos_p, c_pos_p, color=color_poisson, alpha=0.6)
    #plt.scatter(r_neg_p, c_neg_p, color=color_poisson, alpha=0.6)

    #plt.scatter(r_pos_pl, c_pos_pl, color=color_power_law, alpha=0.6) # Power-law positive
    #plt.scatter(r_neg_pl, c_neg_pl, color=color_power_law, alpha=0.6) # Power-law negative
    
p_vals1 = np.linspace(0,0.6,200)

"""for p in p_vals1:
    network_walls = nx.watts_strogatz_graph(100000, 20, p)
    r = nx.degree_assortativity_coefficient(network_walls)
    c = nx.average_clustering(network_walls)
    plt.scatter(r, c, color='gold', alpha=0.1) # Scatter plot for Watts-Strogatz networks"""


plt.xlabel('Assortativity Coefficient (r)')
plt.ylabel('Clustering Coefficient (c)')
plt.title('Assortativity vs Clustering Coefficient for Different Networks for 100,000 Nodes')

# Create custom legend handles
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Poisson', markerfacecolor=color_poisson, markersize=8),
    #Line2D([0], [0], marker='o', color='w', label='Geometric', markerfacecolor=color_geometric, markersize=8,),
    #Line2D([0], [0], marker='o', color='w', label='Power Law', markerfacecolor=color_power_law, markersize=8),
    #Line2D([0], [0], marker='o', color='w', label='Watts-Strogatz', markerfacecolor='gold', markersize=8)
]
plt.legend(handles=legend_elements)

plt.grid(True)
plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.savefig('NNNEWNNassortativity_vs_clustering_scatter.svg', dpi=300) # Save the figure

