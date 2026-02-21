import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def plot_graph(adj: np.ndarray) -> plt.Figure:
    raise NotImplementedError

def plot_degree_distribution(adj: np.ndarray) -> plt.Figure:
    raise NotImplementedError

def plot_centrality(adj: np.ndarray, centrality: dict[int, float]) -> plt.Figure:
    raise NotImplementedError

def plot_transition_heatmap(adj: np.ndarray) -> plt.Figure:
    raise NotImplementedError