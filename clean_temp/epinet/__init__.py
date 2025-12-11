"""epinet package: exposes modules for network epidemic experiments."""

from .network import Network
from .epidemic import simulate_epidemic, length_of_infection_period
from .assortativity import assortative_negative_network

__all__ = [
    'Network',
    'simulate_epidemic',
    'length_of_infection_period',
    'assortative_negative_network',
]
