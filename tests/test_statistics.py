import pytest
from tonnetz.graph.builder import random_adjacency_graph
from tonnetz.graph.statistics import (
    find_degree_distribution,
    find_clustering_coefficient,
    find_average_clustering,
    find_diameter,
    find_giant_component_size,
    print_statistics,
)

def print_test():
    random_graph = random_adjacency_graph()
    print_statistics(random_graph)

if __name__ == "__main__":
    print_test()