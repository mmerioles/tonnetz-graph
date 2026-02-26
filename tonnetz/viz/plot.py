import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def create_note_labels() -> dict:
    """Create a mapping from node indices to note names (0=C2, 47=B5)."""
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return {i: f"{note_names[i % 12]}{i // 12 + 2}" for i in range(48)}


def plot_graph(
    input_graph: nx.DiGraph,
    show_isolated_nodes: bool = False,
    show: bool = True,
    name: str = "Network Graph",
) -> None:

    G = input_graph

    # Remove isolated nodes
    if not show_isolated_nodes:
        isolated = list(nx.isolates(G))
        G.remove_nodes_from(isolated)

    # Create full note labels, but only use those for nodes that exist in G
    all_note_labels = create_note_labels()
    note_labels = {n: all_note_labels[n] for n in G.nodes()}

    node_pos = nx.kamada_kawai_layout(G, scale=5)
    degree_centrality = dict(G.in_degree())
    node_colors = [degree_centrality[n] for n in G.nodes()]
    node_sizes = [max(degree_centrality[n] * 60, 40) for n in G.nodes()]
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    edge_widths = [w * 0.3 for w in edge_weights]

    nx.draw_networkx_edges(
        G,
        node_pos,
        arrows=True,
        connectionstyle="arc3,rad=0",
        arrowsize=3,
        width=edge_widths,
        edge_color="black",
    )
    nx.draw_networkx_nodes(
        G,
        node_pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.Blues,
        alpha=0.95,
    )
    nx.draw_networkx_labels(
        G,
        node_pos,
        labels=note_labels,
        font_color="black",
        font_family="Arial",
        font_size=12,
    )

    plt.title(
        name,
        fontsize=18,
        fontweight="bold",
        fontfamily="Times New Roman",
        color="black",
        pad=20,
    )
    plt.tight_layout()
    if show:
        plt.show()


def plot_degree_distribution(
    degree_distribution: dict[int, float], show: bool = True
) -> None:
    """Plot Histogram of degree distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))
    degrees = list(degree_distribution.keys())
    counts = list(degree_distribution.values())
    ax.bar(degrees, counts, color="blue", alpha=0.7)
    ax.set_xlabel("Degree")
    ax.set_ylabel("Count")
    ax.set_title("Degree Distribution Histogram")
    plt.tight_layout()
    if show:
        plt.show()


def plot_centrality(adj: np.ndarray, centrality: dict[int, float]) -> plt.Figure:
    raise NotImplementedError


def plot_transition_heatmap(adj: np.ndarray) -> plt.Figure:
    raise NotImplementedError
