import numpy as np
import networkx as nx
from tonnetz.util.util import create_note_labels

class Stats:
    @staticmethod
    def find_degree_distribution(adj_matrix: np.ndarray) -> dict[int, float]:
        """
        Finds the degree distribution from a np matrix

        intution: higher degree distribution -> higher likelihood of transitioning to this note (hierchical tonal structure)
        """

        A = (adj_matrix != 0).astype(int)

        out_degree = A.sum(axis=1)
        in_degree  = A.sum(axis=0)

        total_degree = in_degree + out_degree

        if total_degree.size == 0:
            return {}

        max_deg = int(total_degree.max())

        counts = np.bincount(total_degree, minlength=max_deg + 1)
        probs = counts / counts.sum()

        dist = {k: float(probs[k]) for k in range(len(probs)) if counts[k] > 0}

        assert sum(dist.values()) == 1, 'error: degree dist must sum to 1'

        return dist

    @staticmethod
    def find_clustering_coefficient(adj_matrix: np.ndarray) -> dict[int, float]:
        """
        Find clustering coefficients

        i.e - how tightly interconnected a note's neighbors are

        for example: if a note A has edges to neighbors B and C, how likely will a note transition to C given that it transitioned to B

        intuition: if a note has high clustering coefficient, this means notes connected to a given node have high probability of transitioning between each other!
        """
        A = (adj_matrix != 0).astype(int)
        A = ((A + A.T) > 0).astype(int)
        np.fill_diagonal(A, 0)
        k = A.sum(axis=1)
        A3 = A @ A @ A

        clustering = {}

        for i in range(len(A)):
            if k[i] < 2:
                clustering[i] = 0.0
            else:
                clustering[i] = A3[i, i] / (k[i] * (k[i] - 1))

        return clustering

    @staticmethod
    def find_average_clustering(adj_matrix: np.ndarray) -> float:
        """
        Finds overall average clustering coefficient of the graph

        intuition: how harmonically structured is the entire piece?
        """

        clustering = Stats.find_clustering_coefficient(adj_matrix)

        if not clustering:
            return 0.0

        return float(np.mean(list(clustering.values())))

    @staticmethod
    def find_diameter(adj_matrix: np.ndarray) -> int:
        """
        Finds the diameter of the graph "longest shortest path"

        intuition: how wide the musical pitch is of a piece
        """
        
        A = (adj_matrix != 0).astype(int)
        U = ((A + A.T) > 0).astype(int)
        np.fill_diagonal(U, 0)
        Gu = nx.from_numpy_array(U, create_using=nx.Graph())

        if Gu.number_of_nodes() <= 1 or Gu.number_of_edges() == 0:
            return 0

        giant_nodes = max(nx.connected_components(Gu), key=len)
        H = Gu.subgraph(giant_nodes)

        if H.number_of_nodes() <= 1:
            return 0

        return int(nx.diameter(H))

    @staticmethod
    def find_giant_component_size(adj_matrix: np.ndarray) -> int:
        """
        Finds the size of the giant component in a given graph

        intuition: if the giant component is close to 48, then most notes are part of the
                same tonal/transition universe. If it's much smaller, the piece's transitions are
                split into separate pitch regions that don't interact much.
        """

        A = (adj_matrix != 0).astype(int)
        U = ((A + A.T) > 0).astype(int)
        np.fill_diagonal(U, 0)

        Gu = nx.from_numpy_array(U, create_using=nx.Graph())

        if Gu.number_of_nodes() == 0:
            return 0
        if Gu.number_of_edges() == 0:
            # nodes exist but no connections; largest component is any single node
            return 1 if Gu.number_of_nodes() > 0 else 0

        giant_nodes = max(nx.connected_components(Gu), key=len)
        return int(len(giant_nodes))
    
    @staticmethod
    def print_statistics(adj_matrix: np.ndarray) -> None:
        """
        Cleanly prints all stats from a given numpy array 
        """
        width = 40
        print(f"\n{'='*width}")
        print(f"{'Degree Distribution'.center(width)}")
        print(f"{'='*width}")
        deg_dist = Stats.find_degree_distribution(adj_matrix)
        for deg, dist in deg_dist.items():
            print(f"Degree {deg} → Frequency: {dist:.3f}")

        print(f"\n{'='*width}")
        print(f"{'Clustering Coefficients'.center(width)}")
        print(f"{'='*width}")
        clust_coeff = Stats.find_clustering_coefficient(adj_matrix)
        labels = create_note_labels()
        for node, coeff in clust_coeff.items():
            print(f"Node {node} ({labels[node]}) → {coeff:.3f}")

        print(f"\n{'='*width}")
        print(f"{'Average Clustering'.center(width)}")
        print(f"{'='*width}")
        avg_clust = Stats.find_average_clustering(adj_matrix)
        print(f"Average Clustering: {avg_clust:.3f}")

        print(f"\n{'='*width}")
        print(f"{'Diameter'.center(width)}")
        print(f"{'='*width}")
        diam = Stats.find_diameter(adj_matrix)
        print(f"Diameter: {diam}")

        print(f"\n{'='*width}")
        print(f"{'Giant Component Size'.center(width)}")
        print(f"{'='*width}")
        giant_comp_size = Stats.find_giant_component_size(adj_matrix)
        print(f"Giant Component Size: {giant_comp_size}")
