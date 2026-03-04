import os
from tonnetz.midi.parser import gen_transition_poly
from tonnetz.graph.builder import build_graph
from tonnetz.viz.plot import plot_graph, plot_degree_distribution
from tonnetz.graph.statistics import Stats
from tonnetz.graph.centrality import print_centralities
from tonnetz.graph.centrality import get_centralities
from tonnetz.gen.walk import biased_random_walk
from tonnetz.gen.create_midi import create_midi_from_list

filename = "My_Heart_Will_Go_On.mid"
chord_overlay_filename = "My_Heart_Will_Go_On_combined.mid"
# filename = "beethooven-3rd-movement.mid"
channel_number = 0

# Get the project root directory (parent of the scripts directory)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
midi_file = os.path.join(project_root, "raw_midi", filename)
transition_matrix = gen_transition_poly(midi_file, channel_number)

# Generate random-walk melody variants from the selected Tonnetz graph.
raw_midi_dir = os.path.join(project_root, "raw_midi")
melody_outputs: dict[str, str] = {}
# Include original combined melody as a selectable baseline.
melody_outputs[chord_overlay_filename] = chord_overlay_filename
for mode, seed in (("degree", 11), ("betweenness", 29), ("eigenvector", 47)):
    sequence = biased_random_walk(
        transition_matrix,
        length=96,
        rest_prob=0.25,
        centrality_type=mode,
        seed=seed,
    )
    out_name = f"rw_melody_{mode}.mid"
    out_path = os.path.join(raw_midi_dir, out_name)
    create_midi_from_list(
        sequence,
        output_path=out_path,
        bpm=88.0,
        channel=0,
        velocity=92,
        note_length_beats=0.5,
    )
    melody_outputs[out_name] = out_name

# Build the graph and plot it
G = build_graph(transition_matrix)
ctr = get_centralities(transition_matrix)
plot_graph(
    G,
    show_isolated_nodes=False,
    name=filename,
    centralities=ctr,
    overlay_chord_midi_name=chord_overlay_filename,
    overlay_melody_options=melody_outputs,
)


# Compute and print statistics
# Stats.print_statistics(transition_matrix)
# print_centralities(transition_matrix)

# Plot Degree Distribution Histogram
# degree_distribution = Stats.find_degree_distribution(transition_matrix)
# plot_degree_distribution(degree_distribution)

# TODO: Visualisation: display info about MIDI file, statistics
