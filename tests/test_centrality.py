from tonnetz.graph.centrality import find_betweenness_centrality, find_eigenvector_centrality
import numpy as np

x = np.ones((48, 48))
find_eigenvector_centrality(x)
print("hi")
