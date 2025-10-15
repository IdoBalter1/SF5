from Main import Network, naive_create_network, gnp_two_stage, dfs, configuration_model, get_degree_distribution,simulate_epidemic
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from collections import defaultdict
from itertools import combinations
import math
import timeit


#plot the infected counts
def plot_suscetpible_infected_recovered_counts(I_counts,S_counts,R_counts, title):
    plt.figure(figsize=(10, 6))
    plt.plot(I_counts, marker='o', linestyle='-', color='b')
    plt.legend(['Infected'])
    plt.title(title)
    plt.xlabel('Week')
    plt.ylabel('Number of Individuals')
    plt.grid()
    plt.show()

# Example usage
if __name__ == "__main__":
    num_nodes = 10000
    # lam = 0.1 # Infection probability - will be set in the loop
    num_weeks = 50

    # Create a configuration model network with Poisson degree distribution
    # We can create the network once if it's the same for all lambda values,
    # or inside the loop if it needs to be different (e.g., if lambda influenced network structure)
    # For an SIR model, the network structure is usually fixed for a set of simulations.
    print("Creating network...")
    network = Network.from_poisson(20, num_nodes)  # Mean degree of 20
    print("Network created.")

    lambda_values = np.arange(0, 1.05, 0.05) # 0 to 1 inclusive, step 0.05

    for lam in lambda_values:
        print(f"\nRunning simulation for lambda = {lam:.2f}")
        # Randomly select initial infected individuals for each simulation run
        # This ensures each simulation starts with a fresh random set of infected nodes
        initial_infected = random.sample(list(network.adj.keys()), k=20) # Ensure sampling from actual nodes

        # Simulate the epidemic
        S_counts, I_counts, R_counts = simulate_epidemic(network, initial_infected, lam, num_weeks)

        # Plot the results
        plot_title = (f"Network Size: {num_nodes}, Mean Degree: 20\n"
                      f"SIR Simulation: λ = {lam:.2f}, Initial Infected: 20, Weeks: {num_weeks}")
        plot_suscetpible_infected_recovered_counts(I_counts, S_counts, R_counts, plot_title)

    print("\nAll simulations complete.")


    