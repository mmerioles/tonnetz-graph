import numpy as np
import pytest
from tonnetz.graph.centrality import (
    find_betweenness_centrality,
    find_eigenvector_centrality,
    find_degree_centrality,
)

# --- Fixtures ---

@pytest.fixture
def simple_adj():
    """Small deterministic 4-node matrix for predictable results."""
    return np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ], dtype=float)

@pytest.fixture
def random_adj():
    """Mirrors your build_random_adjacency_matrix() output."""
    from tonnetz.graph.builder import build_random_adjacency_matrix
    np.random.seed(42)
    return build_random_adjacency_matrix()


# --- Return type tests ---

def test_betweenness_returns_str_keys(simple_adj):
    result = find_betweenness_centrality(simple_adj)
    assert all(isinstance(k, str) for k in result.keys())

def test_eigenvector_returns_str_keys(simple_adj):
    result = find_eigenvector_centrality(simple_adj)
    assert all(isinstance(k, str) for k in result.keys())

def test_degree_returns_int_keys(simple_adj):
    result = find_degree_centrality(simple_adj)
    assert all(isinstance(k, int) for k in result.keys())


# --- Return value tests ---

def test_all_scores_are_floats(simple_adj):
    for fn in [find_betweenness_centrality, find_eigenvector_centrality, find_degree_centrality]:
        result = fn(simple_adj)
        assert all(isinstance(v, float) for v in result.values())

def test_scores_are_normalized(simple_adj):
    """All centrality scores should be in [0, 1]."""
    for fn in [find_betweenness_centrality, find_eigenvector_centrality, find_degree_centrality]:
        result = fn(simple_adj)
        assert all(0.0 <= v <= 1.0 for v in result.values()), f"Out-of-range score in {fn.__name__}"

def test_node_count_matches(simple_adj):
    """Output should have one entry per node."""
    n = simple_adj.shape[0]
    assert len(find_betweenness_centrality(simple_adj)) == n
    assert len(find_eigenvector_centrality(simple_adj)) == n
    assert len(find_degree_centrality(simple_adj)) == n


# --- Larger random graph ---

def test_random_graph_runs_without_error(random_adj):
    find_betweenness_centrality(random_adj)
    find_eigenvector_centrality(random_adj)
    find_degree_centrality(random_adj)

def test_random_graph_node_count(random_adj):
    n = random_adj.shape[0]
    assert len(find_degree_centrality(random_adj)) == n