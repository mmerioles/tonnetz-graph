import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tonnetz.util.util import create_note_labels
from matplotlib.widgets import RadioButtons

def plot_graph(input_graph: nx.DiGraph, 
               show_isolated_nodes: bool = False, 
               show: bool = True,
               name: str = 'Network Graph',
               centralities: dict = None) -> None:
    
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
    node_artist = nx.draw_networkx_nodes(
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

    # centrality overlay
    if centralities:
        nodes = list(G.nodes())

        deg_norm = centralities["deg"]
        btw_str  = centralities["btw"]  
        eig_str  = centralities["eig"]  

        deg_vals = np.array([deg_norm.get(int(n), 0.0) for n in nodes], dtype=float)
        deg_min = float(deg_vals.min()) if deg_vals.size else 0.0
        deg_max = float(deg_vals.max()) if deg_vals.size else 1.0
        if deg_max == deg_min:
            deg_max = deg_min + 1.0

        btw = {int(k): float(v) for k, v in btw_str.items()}
        eig = {int(k): float(v) for k, v in eig_str.items()}

        def rescale_to_degree_range(vals: np.ndarray) -> np.ndarray:
            if vals.size == 0:
                return vals
            vmin = float(vals.min())
            vmax = float(vals.max())
            if vmax == vmin:
                return np.full(vals.shape, deg_min, dtype=float)
            t = (vals - vmin) / (vmax - vmin)
            return deg_min + t * (deg_max - deg_min)

        def metric_values(metric: str) -> np.ndarray:
            if metric == "degree":
                base = np.array([deg_norm.get(int(n), 0.0) for n in nodes], dtype=float)
                return rescale_to_degree_range(base)
            if metric == "betweenness":
                base = np.array([btw.get(int(n), 0.0) for n in nodes], dtype=float)
                return rescale_to_degree_range(base)
            if metric == "eigenvector":
                base = np.array([eig.get(int(n), 0.0) for n in nodes], dtype=float)
                return rescale_to_degree_range(base)
            raise ValueError(metric)

        def sizes_from(vals: np.ndarray) -> np.ndarray:
            return np.array([max(v * 6000, 40) for v in vals], dtype=float)

        fig = plt.gcf()
        fig.set_size_inches(12, 8, forward=True)

        rax = fig.add_axes([0.02, 0.35, 0.17, 0.25])
        radio = RadioButtons(rax, ("degree", "betweenness", "eigenvector"), active=0)
        rax.set_title("Centrality", fontsize=11)
        def on_change(label: str):
            vals = metric_values(label)
            node_artist.set_array(vals)
            node_artist.set_sizes(sizes_from(vals))
            node_artist.set_clim(deg_min, deg_max)
            fig.canvas.draw_idle()

        radio.on_clicked(on_change)
        on_change("degree")

    if show:
        plt.show()

