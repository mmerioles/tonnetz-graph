import os
from tonnetz.midi.parser import gen_transition_from_mono_midi
from tonnetz.graph.builder import build_graph
from tonnetz.viz.plot import plot_graph
from tonnetz.graph.statistics import Stats
from tonnetz.graph.centrality import print_centralities
filename = "My_Heart_Will_Go_On.mid"
channel_number = 3

# Get the project root directory (parent of the scripts directory)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
midi_file = os.path.join(project_root, "raw_midi", filename)
transition_matrix = gen_transition_from_mono_midi(midi_file, channel_number)

G = build_graph(transition_matrix)
plot_graph(G, show_isolated_nodes=False, show=True, name=filename)

Stats.print_statistics(transition_matrix)
print_centralities(transition_matrix)

# TODO: Visualisation: display info about MIDI file, statistics
