# SF5: Epidemic Network Simulation

A Python package for simulating epidemic spread on various network topologies, with support for different network types, assortativity configurations, and interactive web visualization.

## Overview

This project implements epidemic simulation models on networks, exploring how network structure (degree distribution, assortativity, clustering) affects disease spread dynamics. The simulation uses a Susceptible-Infected-Recovered (SIR) model where nodes can be in one of three states.

## Project Structure

```
SF5/
├── Week 1/                    # Week 1 coursework files
│   ├── clean_temp/           # Refactored code with Flask web app
│   │   ├── epinet/          # Main package
│   │   ├── app.py           # Flask web application
│   │   ├── Main.py          # Command-line entry point
│   │   └── requirements.txt # Dependencies
│   └── epinet/              # Package source code
├── SF5/                      # Additional project files
└── README.md                 # This file
```

## Features

- **Multiple Network Types**: Generate networks with Poisson, Geometric, or Power-law degree distributions
- **Assortativity Control**: Create assortative or disassortative networks to study mixing patterns
- **Epidemic Simulation**: SIR model simulation with configurable infection probabilities
- **Interactive Web Interface**: Flask-based web app for visualizing simulations
- **Network Metrics**: Compute clustering coefficients, assortativity, and other network properties

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Setup

1. Navigate to the project directory:
```powershell
cd "C:\Users\idoba\OneDrive\Documents\IIA\Coursework\SF5\Week 1\clean_temp"
```

2. Install dependencies:
```powershell
python -m pip install -r requirements.txt
```

Required packages:
- `numpy` - Numerical computations
- `matplotlib` - Plotting and visualization
- `scipy` - Scientific computing (disjoint sets, statistics)
- `networkx` - Network analysis
- `flask` - Web framework for interactive interface
- `gunicorn` - WSGI server (for deployment)

## Usage

### Command Line

Run a simple simulation from the command line:

```powershell
cd "Week 1\clean_temp"
python Main.py
```

### Web Interface

Launch the interactive web application:

```powershell
cd "Week 1\clean_temp"
python app.py
```

Then open your browser to `http://127.0.0.1:5000` to access the interactive simulation interface.

### As a Package

To use `epinet` as a package in other projects, install it in editable mode:

```powershell
cd "Week 1\clean_temp"
python -m pip install -e .
```

## Package Modules

The `epinet` package consists of the following modules:

- **`network.py`**: `Network` class and graph generation methods (Poisson, Geometric, Power-law)
- **`epidemic.py`**: Epidemic simulation functions and SIR model implementation
- **`assortativity.py`**: Functions to create assortative/disassortative networks
- **`metrics.py`**: Network metrics, vaccination strategies, and probability computations
- **`plotting.py`**: Visualization helpers and plotting routines

## Example

```python
from epinet.network import Network
from epinet.epidemic import simulate_epidemic
import random

# Create a Poisson network with 200 nodes and average degree 4
network = Network.from_poisson(avg_degree=4.0, num_nodes=200)

# Select 3 random initial infected nodes
initial_infected = random.sample(network.node_list(), 3)

# Simulate epidemic with infection probability 0.05
susceptible, infected, recovered = simulate_epidemic(
    network, 
    initial_infected, 
    infection_prob=0.05, 
    num_weeks=50
)

# Plot results
import matplotlib.pyplot as plt
weeks = range(len(susceptible))
plt.plot(weeks, susceptible, label='Susceptible')
plt.plot(weeks, infected, label='Infected')
plt.plot(weeks, recovered, label='Recovered')
plt.legend()
plt.show()
```

## Network Types

- **Poisson**: Random network with Poisson degree distribution (Erdős–Rényi-like)
- **Geometric**: Random geometric graph with spatial structure
- **Power-law**: Scale-free network with power-law degree distribution

## Assortativity

The package supports creating networks with different assortativity patterns:
- **Assortative**: High-degree nodes tend to connect to other high-degree nodes
- **Disassortative**: High-degree nodes tend to connect to low-degree nodes
- **Neutral**: Random mixing (default)

## Coursework

This project is part of coursework for IIA (presumably an Information/Network Analysis course). The repository includes:
- Interim reports documenting progress
- Final report with figures
- Experimental results and analysis

## Development

The codebase has been refactored from a single large script into a modular package structure for better maintainability. The main entry points are:
- `Main.py` - Simple command-line demonstration
- `app.py` - Interactive web interface

## License

This is coursework material. Please refer to your institution's academic policies regarding code sharing and collaboration.

## Notes

- If your editor shows "unresolved import" warnings, ensure the project root is in your Python path
- Running scripts from the package root directory works because Python adds the current directory to `sys.path`
- For large networks (>5000 nodes), the web interface may sample nodes for visualization performance

