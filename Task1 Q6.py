from Main import Network,gnp_two_stage,naive_create_network
import numpy as np
import random
from itertools import combinations
from collections import defaultdict
import matplotlib.pyplot as plt
import timeit

time_list_naive = []
time_list_sample = []
n_values = [2**x for x in range(6, 10)]
for n in n_values:
    num_nodes = n
    p = 10/(num_nodes-1)
    timeit_naive = timeit.timeit(lambda: naive_create_network(num_nodes, p), number=100)
    time_list_naive.append(timeit_naive)
    timeit_sample = timeit.timeit(lambda: gnp_two_stage(num_nodes, p), number=100)
    time_list_sample.append(timeit_sample)
    print(f"Number of nodes: {num_nodes}, Naive time:'timeit_naive:.4f' seconds, Sample time: {timeit_sample:.4f} seconds")

log_n = np.log10(n_values)
log_naive = np.log10(time_list_naive)
log_sample = np.log10(time_list_sample)
slope_naive, _ = np.polyfit(log_n, log_naive, 1)
slope_sample, _ = np.polyfit(log_n, log_sample, 1)

print(f"\nEstimated slope (Naive): {slope_naive:.2f}")
print(f"Estimated slope (Sample-based): {slope_sample:.2f}")
x_vals = [2**x for x in range(5, 11)]
plt.figure(figsize=(8, 5))
plt.loglog(n_values, time_list_naive, 'o-', label="Naive G(n, p)", base=10)
plt.loglog(n_values, time_list_sample, 's-', label="Sample-based G(n, p)", base=10)

plt.xlabel("Number of nodes (n)")
plt.ylabel("Total time (100 runs)")
plt.title("Log-Log Plot of Runtime vs Number of Nodes")
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.show()


