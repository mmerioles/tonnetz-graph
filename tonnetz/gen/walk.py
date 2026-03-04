import numpy as np
import networkx as nx

from tonnetz.graph.builder import build_graph
from tonnetz.graph.centrality import find_eigenvector_centrality

REST = -1
REST_PROB = 0.3
WALK_PROB = 0.7
SEQUENCE_LENGTH = 30


def biased_random_walk(
    adj_matrix: np.ndarray,
    start_node: int | None = None,
    length: int = SEQUENCE_LENGTH,
    rest_prob: float = REST_PROB,
    seed: int | None = None,
) -> list[int]:
    """
    Generate a sequence of notes by performing a biased random walk
    on the Tonnetz graph.

    At each step:
      - With probability `rest_prob` (0.3): emit a rest (-1) and stay on current node
      - With probability `1 - rest_prob` (0.7): move to a neighbor, biased by
        edge weight * eigenvector centrality of the neighbor

    Parameters
    ----------
    adj_matrix : np.ndarray
        Square weighted adjacency matrix (n x n), nodes in range [0, n-1].
    start_node : int | None
        Starting node. If None, picks the highest eigenvector centrality node.
    length : int
        Length of the output sequence (default 30).
    rest_prob : float
        Probability of emitting a rest at each step (default 0.3).
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    list[int]
        Sequence of note indices [0, n-1] and rests (-1), of fixed length `length`.
        Example: [2, 5, -1, 10, 46, -1, 3, ...]
    """
    rng = np.random.default_rng(seed)

    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph())

    # Eigenvector centrality as bias — higher centrality = more attractive node
    eigenvector = find_eigenvector_centrality(adj_matrix)
    centrality = {int(k): v for k, v in eigenvector.items()}

    # Default start: node with highest eigenvector centrality
    if start_node is None:
        start_node = max(centrality, key=centrality.get)

    sequence: list[int] = []
    current = start_node

    while len(sequence) < length:
        # 0.3 chance of rest
        if rng.random() < rest_prob:
            sequence.append(REST)
            continue  # stay on current node, don't move

        # 0.7 chance of walking — get neighbors and their edge weights
        neighbors = list(G.successors(current))

        if not neighbors:
            # Dead end — emit rest and jump to highest centrality node
            sequence.append(REST)
            current = max(centrality, key=centrality.get)
            continue

        # Bias transition probability by edge_weight * neighbor centrality
        weights = np.array([
            G[current][nb]["weight"] * centrality[nb]
            for nb in neighbors
        ])
        weights = weights / weights.sum()  # normalize to probabilities

        current = int(rng.choice(neighbors, p=weights))
        sequence.append(current)

    return sequence


def print_sequence(sequence: list[int]) -> None:
    """Pretty-print the generated sequence."""
    print(f"\nGenerated sequence (length={len(sequence)}):")
    print(sequence)
    notes = [str(n) if n != REST else "R" for n in sequence]
    print(" → ".join(notes))


if __name__ == "__main__":
    n = 48
    mat = np.random.exponential(0.3, size=(n, n))
    mat = mat / mat.max()
    mat[mat < 0.2] = 0

    sequence = biased_random_walk(mat, seed=42)
    print_sequence(sequence)