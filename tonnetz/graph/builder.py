import numpy as np
import networkx as nx


def build_graph(adj: np.ndarray) -> nx.DiGraph:
    G=nx.from_numpy_array(adj, create_using=nx.DiGraph())
    return G
    
def build_random_adjacency_matrix(n: int = 48) -> np.ndarray:
    mat=np.random.exponential(0.3,size=(n,n))
    mat=mat/mat.max()
    mat[mat<0.2]=0
    return mat
