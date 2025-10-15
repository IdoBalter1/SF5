"""Small entrypoint demonstrating the refactored modules.

Run this script from the folder to exercise imports and a tiny example.
"""

import numpy as np

from epinet.network import Network
from epinet.assortativity import assortative_negative_network


def main():
    np.random.seed(42)
    net, _ = assortative_negative_network(Network.from_poisson(6, 300))
    print('Created an assortative-negative sample network with', net.edge_count, 'edges')


if __name__ == '__main__':
    main()