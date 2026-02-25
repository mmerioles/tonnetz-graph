import numpy as np
from builder import random_adjacency_graph
import networkx as nx
from pprint import pprint

def find_degree_distribution(adj_matrix: np.ndarray) -> dict[int, float]:
    """
    Finds the degree distribution from a np matrix

    intution: higher degree distribution -> higher likelihood of transitioning to this note (hierchical tonal structure)
    """

    A = (adj_matrix != 0).astype(int)

    out_degree = A.sum(axis=1)
    in_degree  = A.sum(axis=0)

    total_degree = in_degree + out_degree

    if total_degree.size == 0:
        return {}

    max_deg = int(total_degree.max())

    counts = np.bincount(total_degree, minlength=max_deg + 1)
    probs = counts / counts.sum()

    dist = {k: float(probs[k]) for k in range(len(probs)) if counts[k] > 0}

    # sanity check print if all items sum to 1
    print(f'{sum(dist.values())=}')

    return dist

def find_clustering_coefficient(adj_matrix: np.ndarray) -> dict[int, float]:
    """
    Find clustering coefficients

    i.e - how tightly interconnected a note's neighbors are

    for example: if a note A has edges to neighbors B and C, how likely will a note transition to C given that it transitioned to B

    intuition: if a note has high clustering coefficient, this means notes connected to a given node have high probability of transitioning between each other!
    """
    A = (adj_matrix != 0).astype(int)
    A = ((A + A.T) > 0).astype(int)
    np.fill_diagonal(A, 0)
    k = A.sum(axis=1)
    A3 = A @ A @ A

    clustering = {}

    for i in range(len(A)):
        if k[i] < 2:
            clustering[i] = 0.0
        else:
            clustering[i] = A3[i, i] / (k[i] * (k[i] - 1))

    return clustering

def find_average_clustering(adj_matrix: np.ndarray) -> float:
    """
    Finds overall average clustering coefficient of the graph

    intuition: how harmonically structured is the entire piece?
    """

    clustering = find_clustering_coefficient(adj_matrix)

    if not clustering:
        return 0.0

    return float(np.mean(list(clustering.values())))

def find_diameter(adj_matrix: np.ndarray) -> int:
    """
    Finds the diameter of the graph "longest shortest path"

    intuition: how wide the musical pitch is of a piece
    """
    
    A = (adj_matrix != 0).astype(int)
    U = ((A + A.T) > 0).astype(int)
    np.fill_diagonal(U, 0)
    Gu = nx.from_numpy_array(U, create_using=nx.Graph())

    if Gu.number_of_nodes() <= 1 or Gu.number_of_edges() == 0:
        return 0

    giant_nodes = max(nx.connected_components(Gu), key=len)
    H = Gu.subgraph(giant_nodes)

    if H.number_of_nodes() <= 1:
        return 0

    return int(nx.diameter(H))

def find_giant_component_size(adj_matrix: np.ndarray) -> int:
    """
    Finds the size of the giant component in a given graph

    intuition: if the giant component is close to 48, then most notes are part of the
            same tonal/transition universe. If it's much smaller, the piece's transitions are
            split into separate pitch regions that don't interact much.
    """

    A = (adj_matrix != 0).astype(int)
    U = ((A + A.T) > 0).astype(int)
    np.fill_diagonal(U, 0)

    Gu = nx.from_numpy_array(U, create_using=nx.Graph())

    if Gu.number_of_nodes() == 0:
        return 0
    if Gu.number_of_edges() == 0:
        # nodes exist but no connections; largest component is any single node
        return 1 if Gu.number_of_nodes() > 0 else 0

    giant_nodes = max(nx.connected_components(Gu), key=len)
    return int(len(giant_nodes))

if __name__ == "__main__":
    mat=random_adjacency_graph()
    pprint(find_degree_distribution(mat))
    pprint(find_clustering_coefficient(mat))
    print(find_diameter(mat))
    print(find_giant_component_size(mat))
