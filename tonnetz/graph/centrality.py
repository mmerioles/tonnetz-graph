import numpy as np
import networkx as nx


def find_betweenness_centrality(adj_matrix: np.ndarray) -> dict[str, float]:
    """
    Compute betweenness centrality for each node in the graph
    represented by adj_matrix.

    Parameters
    ----------
    adj_matrix : np.ndarray
        Square weighted adjacency matrix (n x n).

    Returns
    -------
    dict[str, float]
        Mapping of node label (as str) -> normalized betweenness score.
    """
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph())
    scores = nx.betweenness_centrality(G, normalized=True, weight="weight")
    return {str(node): score for node, score in scores.items()}


def find_eigenvector_centrality(adj_matrix: np.ndarray) -> dict[str, float]:
    """
    Compute eigenvector centrality for each node in the graph
    represented by adj_matrix.

    Falls back to the exact numpy solver if power iteration fails to
    converge (common with sparse or disconnected graphs).

    Parameters
    ----------
    adj_matrix : np.ndarray
        Square weighted adjacency matrix (n x n).

    Returns
    -------
    dict[str, float]
        Mapping of node label (as str) -> eigenvector centrality score.
    """
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph())
    try:
        scores = nx.eigenvector_centrality(
            G, max_iter=1000, tol=1e-6, weight="weight"
        )
    except nx.PowerIterationFailedConvergence:
        scores = nx.eigenvector_centrality_numpy(G, weight="weight")
    return {str(node): score for node, score in scores.items()}


def find_degree_centrality(adj_matrix: np.ndarray) -> dict[int, float]:
    """
    Compute in-degree centrality for each node in the graph
    represented by adj_matrix.

    Parameters
    ----------
    adj_matrix : np.ndarray
        Square weighted adjacency matrix (n x n).

    Returns
    -------
    dict[int, float]
        Mapping of node index (int) -> normalized in-degree centrality score.
    """
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph())
    scores = nx.in_degree_centrality(G)
    return {int(node): score for node, score in scores.items()}


def print_top(label: str, scores: dict, top_n: int = 10) -> None:
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    print(f"\n{'='*40}")
    print(f"  {label} — Top {top_n} Nodes")
    print(f"{'='*40}")
    for rank, (node, score) in enumerate(ranked, start=1):
        print(f"  {rank:>2}. Node {str(node):>3}  →  {score:.6f}")

def print_centralities(adj_matrix: np.ndarray) -> None:
    print("\nComputing centralities on a 48-node Tonnetz graph...")

    print_top("Degree (in-degree) Centrality", find_degree_centrality(adj_matrix))
    print_top("Betweenness Centrality",        find_betweenness_centrality(adj_matrix))
    print_top("Eigenvector Centrality",        find_eigenvector_centrality(adj_matrix))