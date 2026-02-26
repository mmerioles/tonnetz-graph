import pytest
import numpy as np
from tonnetz.graph.builder import random_adjacency_graph
from tonnetz.graph.statistics import (
    find_degree_distribution,
    find_clustering_coefficient,
    find_average_clustering,
    find_diameter,
    find_giant_component_size,
    print_statistics,
)

@pytest.fixture
def random_adj():
    np.random.seed(42)
    return random_adjacency_graph()

def test_compile(random_adj):
    find_degree_distribution(random_adj)
    find_clustering_coefficient(random_adj)
    find_average_clustering(random_adj)
    find_diameter(random_adj)
    find_giant_component_size(random_adj)

def test_print():
    random_adj = random_adjacency_graph()
    print_statistics(random_adj)

if __name__ == "__main__":
    test_print()