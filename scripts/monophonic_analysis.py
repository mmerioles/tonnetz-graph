import os
from tonnetz.midi.parser import gen_transition_from_mono_midi
from tonnetz.graph.builder import build_graph
from tonnetz.viz.plot import plot_graph

filename = "My_Heart_Will_Go_On.mid"
channel_number = 3

# Get the project root directory (parent of the scripts directory)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
midi_file = os.path.join(project_root, "raw_midi", filename)
transition_matrix = gen_transition_from_mono_midi(midi_file, channel_number)

G = build_graph(transition_matrix)
plot_graph(G, show_isolated_nodes=False, show=True, name=filename)

# TODO: Visualisation: display info about MIDI file, statistics
