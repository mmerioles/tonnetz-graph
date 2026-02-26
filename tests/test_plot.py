import pytest
from tonnetz.graph.builder import build_random_adjacency_matrix, build_graph
from tonnetz.viz.plot import plot_graph

@pytest.fixture
def random_adj():
    return build_random_adjacency_matrix()

def test_compile(random_adj):
    G = build_graph(random_adj)
    plot_graph(G)

if __name__ == "__main__":
    random_adj = build_random_adjacency_matrix()
    G = build_graph(random_adj)
    plot_graph(G, show=True)