"""Lightweight Flask app to visualize a simple epidemic spread simulation."""

from __future__ import annotations

import random
from typing import Any, Dict

import numpy as np
from flask import Flask, jsonify, render_template, request
import networkx as nx

from epinet.epidemic import simulate_epidemic
from epinet.network import Network

app = Flask(__name__, static_folder="static", template_folder="templates")


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _build_network(num_nodes: int, avg_degree: float, net_type: str) -> Network:
    # Poisson degree distribution gives us an easy knob for expected degree.
    net_type = (net_type or "poisson").lower()
    if net_type == "geometric":
        # For geometric, mean degree ≈ (1-p)/p -> p = 1/(mean+1)
        p = max(1e-3, min(0.99, 1.0 / (avg_degree + 1.0)))
        return Network.from_geometric(p, num_nodes)
    if net_type == "powerlaw":
        alpha = 2.5
        min_deg = 1
        max_deg = max(5, int(avg_degree * 3))
        return Network.from_power_law(alpha, num_nodes, min_deg, max_deg)
    # default: poisson
    return Network.from_poisson(avg_degree, num_nodes)


def _apply_assortativity_bias(network: Network, bias: float) -> Network:
    """Rudimentary bias: positive -> assortative, negative -> disassortative."""
    try:
        from epinet.assortativity import assortative_network, assortative_negative_network
    except Exception:
        return network

    if bias > 0.1:
        new_net, _ = assortative_network(network)
        return new_net
    if bias < -0.1:
        new_net, _ = assortative_negative_network(network)
        return new_net
    return network


def _graph_layout(network: Network, max_nodes: int = 300):
    """Return positions and edges for visualization (normalized 0..1) using circle layout."""
    node_list = network.node_list()
    if len(node_list) > max_nodes:
        sampled = set(random.sample(node_list, max_nodes))
        nodes = list(sampled)
        edges = [(u, v) for u, v in network.edge_list() if u in sampled and v in sampled]
    else:
        nodes = node_list
        edges = network.edge_list()

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    pos = nx.circular_layout(G) if nodes else {}

    # normalize coords to 0..1
    if pos:
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        range_x = max(max_x - min_x, 1e-6)
        range_y = max(max_y - min_y, 1e-6)
        norm_pos = {
            n: ((pos[n][0] - min_x) / range_x, (pos[n][1] - min_y) / range_y) for n in pos
        }
    else:
        norm_pos = {}

    return {
        "nodes": [{"id": int(n), "x": norm_pos[n][0], "y": norm_pos[n][1]} for n in norm_pos],
        "edges": [{"source": int(u), "target": int(v)} for u, v in edges],
    }


def _assortativity(network: Network) -> float:
    G = nx.Graph()
    G.add_nodes_from(range(network.num_nodes))
    G.add_edges_from(network.edge_list())
    if G.number_of_edges() == 0:
        return 0.0
    coeff = nx.degree_assortativity_coefficient(G)
    # guard against nan for degenerate graphs
    return float(0.0 if np.isnan(coeff) else coeff)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/simulate", methods=["POST"])
def simulate() -> Any:
    payload: Dict[str, Any] = request.get_json(force=True, silent=True) or {}

    num_nodes = min(5000, max(10, _safe_int(payload.get("numNodes"), 200)))
    avg_degree = max(0.1, _safe_float(payload.get("avgDegree"), 4.0))
    infection_prob = min(max(_safe_float(payload.get("infectionProb"), 0.05), 0.0), 1.0)
    initial_infected = _safe_int(payload.get("initialInfected"), 3)
    net_type = str(payload.get("networkType", "poisson")).lower()
    assort_bias = float(payload.get("assortativityBias", 0.0))
    assort_bias = max(-1.0, min(1.0, assort_bias))

    if initial_infected < 1:
        initial_infected = 1
    if initial_infected > num_nodes:
        initial_infected = num_nodes

    net = _build_network(num_nodes, avg_degree, net_type)
    net = _apply_assortativity_bias(net, assort_bias)
    all_nodes = net.node_list()
    initial = random.sample(all_nodes, initial_infected)

    # Run until convergence; cap by a generous step limit to avoid infinite loops.
    max_steps = 1000
    susceptible, infected, recovered = simulate_epidemic(net, initial, infection_prob, max_steps)
    week_axis = list(range(len(susceptible)))
    assort = _assortativity(net)
    layout = _graph_layout(net)

    return jsonify(
        {
            "weeks": week_axis,
            "susceptible": susceptible,
            "infected": infected,
            "recovered": recovered,
            "graph": layout,
            "meta": {
                "numNodes": num_nodes,
                "avgDegree": avg_degree,
                "infectionProb": infection_prob,
                "weeksSimulated": len(week_axis) - 1,
                "initialInfected": len(initial),
                "networkType": net_type,
                "assortativity": assort,
                "assortativityBias": assort_bias,
            },
        }
    )


if __name__ == "__main__":
    app.run(debug=True)

