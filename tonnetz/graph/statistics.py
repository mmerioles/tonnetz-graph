import numpy as np


def find_degree_distribution(adj: np.ndarray) -> dict[int, float]:
    raise NotImplementedError

def find_clustering_coefficient(adj: np.ndarray) -> dict[int, float]:
    raise NotImplementedError

def find_average_clustering(adj: np.ndarray) -> float:
    raise NotImplementedError

def find_diameter(adj: np.ndarray) -> int:
    raise NotImplementedError

def find_giant_component_size(adj: np.ndarray) -> int:
    raise NotImplementedError