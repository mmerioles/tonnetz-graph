import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def build_graph(adj: np.ndarray) -> nx.DiGraph:

    G=nx.from_numpy_array(adj, create_using=nx.DiGraph())
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
    plt.show()

    return G
    



def random_adjacency_graph(n: int = 48) -> np.ndarray:
    
    mat=np.random.exponential(0.3,size=(n,n))
    mat=mat/mat.max()
    mat[mat<0.2]=0

    return mat




mat=random_adjacency_graph()
G=build_graph(mat) 
 