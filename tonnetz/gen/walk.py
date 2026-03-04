import numpy as np
import networkx as nx

from tonnetz.graph.centrality import (
    find_degree_centrality,
    find_betweenness_centrality,
    find_eigenvector_centrality,
)

REST = -1
REST_PROB = 0.3
WALK_PROB = 0.7
SEQUENCE_LENGTH = 30
NUM_NOTES = 48


def biased_random_walk(
    adj_matrix: np.ndarray,
    start_node: int | None = None,
    length: int = SEQUENCE_LENGTH,
    rest_prob: float = REST_PROB,
    centrality_type: str = "eigenvector",
    seed: int | None = None,
) -> list[int]:
    """
    Generate a sequence of notes by performing a biased random walk
    on the Tonnetz graph.

    At each step:
      - With probability `rest_prob`: emit a rest (-1) and stay on current node
      - With probability `1 - rest_prob`: move to a neighbor, biased by
        edge weight * selected centrality of the neighbor

    Parameters
    ----------
    adj_matrix : np.ndarray
        Square weighted adjacency matrix (n x n), nodes in range [0, n-1].
    start_node : int | None
        Starting node. If None, picks the highest centrality node for the selected metric.
    length : int
        Length of the output sequence (default 30).
    rest_prob : float
        Probability of emitting a rest at each step (default 0.3).
    centrality_type : str
        Centrality metric used for bias.
        One of: "eigenvector", "betweenness", "degree".
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    list[int]
        Sequence of note indices [0, n-1] and rests (-1), of fixed length `length`.
    """
    rng = np.random.default_rng(seed)
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph())

    ctype = centrality_type.strip().lower()
    if ctype in ("eig", "eigenvector"):
        scores = find_eigenvector_centrality(adj_matrix)
    elif ctype in ("btw", "betweenness"):
        scores = find_betweenness_centrality(adj_matrix)
    elif ctype in ("deg", "degree"):
        scores = find_degree_centrality(adj_matrix)
    else:
        raise ValueError(
            "centrality_type must be one of: 'eigenvector', 'betweenness', 'degree'"
        )

    centrality = {int(k): float(v) for k, v in scores.items()}

    if start_node is None:
        start_node = max(centrality, key=centrality.get)

    sequence: list[int] = []
    current = int(start_node)

    while len(sequence) < length:
        if rng.random() < rest_prob:
            sequence.append(REST)
            continue

        neighbors = list(G.successors(current))
        if not neighbors:
            sequence.append(REST)
            current = max(centrality, key=centrality.get)
            continue

        weights = np.array(
            [G[current][nb]["weight"] * centrality.get(nb, 0.0) for nb in neighbors],
            dtype=float,
        )
        total = float(weights.sum())
        if total <= 0:
            weights = np.full(len(neighbors), 1.0 / len(neighbors), dtype=float)
        else:
            weights = weights / total

        current = int(rng.choice(neighbors, p=weights))
        sequence.append(current)

    return sequence


def purely_random_sequence(
    length: int = SEQUENCE_LENGTH,
    rest_prob: float = REST_PROB,
    seed: int | None = None,
    num_notes: int = NUM_NOTES,
) -> list[int]:
    """
    Generate a sequence by uniform random sampling over note indices
    [0, num_notes-1], with rests (-1) occurring with probability rest_prob.
    """
    if length < 0:
        raise ValueError("length must be >= 0")
    if not (0.0 <= rest_prob <= 1.0):
        raise ValueError("rest_prob must be in [0, 1]")
    if num_notes <= 0:
        raise ValueError("num_notes must be > 0")

    rng = np.random.default_rng(seed)
    out: list[int] = []
    for _ in range(length):
        if rng.random() < rest_prob:
            out.append(REST)
        else:
            out.append(int(rng.integers(0, num_notes)))
    return out


def print_sequence(sequence: list[int]) -> None:
    """Pretty-print the generated sequence."""
    print(f"\nGenerated sequence (length={len(sequence)}):")
    print(sequence)
    notes = [str(n) if n != REST else "R" for n in sequence]
    print(" -> ".join(notes))


if __name__ == "__main__":
    n = 48
    mat = np.random.exponential(0.3, size=(n, n))
    mat = mat / mat.max()
    mat[mat < 0.2] = 0

    for mode in ("degree", "betweenness", "eigenvector"):
        sequence = biased_random_walk(mat, seed=42, centrality_type=mode)
        print(f"\nCentrality = {mode}")
        print_sequence(sequence)
