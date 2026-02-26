import os
from tonnetz.midi.parser import gen_transition_poly
from tonnetz.graph.builder import build_graph
from tonnetz.viz.plot import plot_graph, plot_degree_distribution
from tonnetz.graph.statistics import Stats
from tonnetz.graph.centrality import print_centralities

filename = "My_Heart_Will_Go_On.mid"
channel_number = 0

# Get the project root directory (parent of the scripts directory)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
midi_file = os.path.join(project_root, "raw_midi", filename)
transition_matrix = gen_transition_poly(midi_file, channel_number)

# Build the graph and plot it
G = build_graph(transition_matrix)
plot_graph(G, show_isolated_nodes=False, show=True, name=filename)

# Compute and print statistics
Stats.print_statistics(transition_matrix)
print_centralities(transition_matrix)

# Plot Degree Distribution Histogram
degree_distribution = Stats.find_degree_distribution(transition_matrix)
plot_degree_distribution(degree_distribution)

# TODO: Visualisation: display info about MIDI file, statistics
