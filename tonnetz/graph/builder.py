import numpy as np
import networkx as nx


def build_graph(adj: np.ndarray) -> nx.DiGraph:
    raise NotImplementedError

def random_adjacency(n: int = 48, seed: int = 42) -> np.ndarray:
    raise NotImplementedError