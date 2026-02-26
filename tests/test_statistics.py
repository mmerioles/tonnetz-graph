import pytest

from tonnetz.graph.builder import build_random_adjacency_matrix
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
    return build_random_adjacency_matrix()

def test_compile(random_adj):
    find_degree_distribution(random_adj)
    find_clustering_coefficient(random_adj)
    find_average_clustering(random_adj)
    find_diameter(random_adj)
    find_giant_component_size(random_adj)

def test_print():
    random_adj = build_random_adjacency_matrix()
    print_statistics(random_adj)

if __name__ == "__main__":
    test_print()