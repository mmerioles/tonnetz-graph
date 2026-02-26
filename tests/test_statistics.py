import pytest

from tonnetz.graph.builder import build_random_adjacency_matrix
from tonnetz.graph.statistics import Stats

@pytest.fixture
def random_adj():
    return build_random_adjacency_matrix()

def test_compile(random_adj):
    Stats.find_degree_distribution(random_adj)
    Stats.find_clustering_coefficient(random_adj)
    Stats.find_average_clustering(random_adj)
    Stats.find_diameter(random_adj)
    Stats.find_giant_component_size(random_adj)

def test_print():
    random_adj = build_random_adjacency_matrix()
    Stats.print_statistics(random_adj)

if __name__ == "__main__":
    test_print()