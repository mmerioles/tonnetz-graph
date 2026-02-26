import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def create_note_labels() -> dict:
    """Create a mapping from node indices to note names (0=C2, 47=B5)."""
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return {i: f"{note_names[i % 12]}{i // 12 + 2}" for i in range(48)}


def build_graph(adj: np.ndarray, show_isolated_nodes: bool = False) -> nx.DiGraph:

    G = nx.from_numpy_array(adj, create_using=nx.DiGraph())

    # Remove isolated nodes
    if not show_isolated_nodes:
        isolated = list(nx.isolates(G))
        print(f"Removing {len(isolated)} isolated nodes: {isolated}")
        G.remove_nodes_from(isolated)
        print(
            f"Graph now has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges"
        )

    node_pos = nx.kamada_kawai_layout(G, scale=5)
    degree_centrality = dict(G.in_degree())
    node_colors = [degree_centrality[n] for n in G.nodes()]
    node_sizes = [max(degree_centrality[n] * 40, 30) for n in G.nodes()]
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    edge_widths = [w * 0.3 for w in edge_weights]

    # Create full note labels, but only use those for nodes that exist in G
    all_note_labels = create_note_labels()
    note_labels = {n: all_note_labels[n] for n in G.nodes()}

    nx.draw_networkx_edges(
        G,
        node_pos,
        arrows=True,
        connectionstyle="arc3,rad=0",
        arrowsize=1,
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
        font_size=5,
    )

    plt.title(
        "Network Graph",
        fontsize=18,
        fontweight="bold",
        fontfamily="Times New Roman",
        color="black",
        pad=20,
    )
    plt.tight_layout()
    plt.show()

    return G


def random_adjacency_graph(n: int = 48) -> np.ndarray:

    mat = np.random.exponential(0.3, size=(n, n))
    mat = mat / mat.max()
    mat[mat < 0.2] = 0

    return mat


if __name__ == "__main__":
    mat = random_adjacency_graph()
    G = build_graph(mat)
