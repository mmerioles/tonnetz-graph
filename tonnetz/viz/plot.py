import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def plot_graph(input_graph: nx.DiGraph, show: bool = False) -> None:
    G = input_graph
    node_pos=nx.kamada_kawai_layout(G,scale=5)
    degree_centrality=dict(G.in_degree())
    node_colors=[degree_centrality[n] for n in G.nodes()]
    node_sizes=[max(degree_centrality[n]*40,30) for n in G.nodes()]
    edge_weights=[G[u][v]['weight'] for u,v in G.edges()]
    edge_widths=[w*0.3 for w in edge_weights]

    nx.draw_networkx_edges(G,node_pos,arrows=True,connectionstyle='arc3,rad=0',arrowsize=1,width=edge_widths,edge_color='black')
    nx.draw_networkx_nodes(G,node_pos,node_size=node_sizes,node_color=node_colors,cmap=plt.cm.Blues,alpha=0.95)
    nx.draw_networkx_labels(G,node_pos,font_color='black', font_family='Arial', font_size=5)

    plt.title('Network Graph', fontsize=18, fontweight='bold',fontfamily='Times New Roman', color='black', pad=20)
    plt.tight_layout()
    if show:
        plt.show()

def plot_degree_distribution(adj: np.ndarray) -> plt.Figure:
    raise NotImplementedError

def plot_centrality(adj: np.ndarray, centrality: dict[int, float]) -> plt.Figure:
    raise NotImplementedError

def plot_transition_heatmap(adj: np.ndarray) -> plt.Figure:
    raise NotImplementedError