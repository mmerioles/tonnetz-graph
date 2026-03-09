import os
import csv
import random

from tonnetz.midi.parser import gen_transition_poly
from tonnetz.graph.builder import build_graph
from tonnetz.viz.plot import plot_graph, plot_degree_distribution
from tonnetz.graph.statistics import Stats
from tonnetz.graph.centrality import print_centralities
from tonnetz.graph.centrality import get_centralities
from tonnetz.gen.walk import biased_random_walk, purely_random_sequence
from tonnetz.gen.create_midi import create_midi_from_list

filename = "My_Heart_Will_Go_On.mid"
chord_overlay_filename = "My_Heart_Will_Go_On_combined.mid"
# filename = "beethooven-3rd-movement.mid"
channel_number = 0
ENABLE_OVERLAY = True  # Set True to enable playback overlay (it works visually even if you dont have fluidsynth installed)
LSTM_SEQUENCE_COUNT = 3

# For adding CSV formatted sequences
def load_random_sequences(csv_path: str, count: int) -> list[tuple[int, list[int]]]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No generated sequences found in {csv_path}")

    first_row = rows[0]
    data_rows = rows[1:] if first_row and first_row[0].startswith("step_") else rows
    if len(data_rows) < count:
        raise ValueError(f"Expected at least {count} generated sequences in {csv_path}")

    rng = random.Random()
    selected_indices = rng.sample(range(len(data_rows)), count)
    return [
        (
            seq_idx,
            [int(note) for note in data_rows[seq_idx] if str(note).strip()],
        )
        for seq_idx in selected_indices
    ]

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

# Also add a purely random baseline melody (0..47 with rests=-1).
random_seq = purely_random_sequence(
    length=96,
    rest_prob=0.25,
    seed=101,
    num_notes=48,
)
random_name = "rw_melody_random.mid"
create_midi_from_list(
    random_seq,
    output_path=os.path.join(raw_midi_dir, random_name),
    bpm=88.0,
    channel=0,
    velocity=92,
    note_length_beats=0.5,
)
melody_outputs[random_name] = random_name


# Add three randomly selected LSTM/generated sequences from the dataset CSV.
generated_seq_csv = os.path.join(project_root, "data", "sequences.csv")
if os.path.exists(generated_seq_csv):
    for idx, (selected_seq_idx, generated_sequence) in enumerate(
        load_random_sequences(generated_seq_csv, LSTM_SEQUENCE_COUNT),
        start=1,
    ):
        output_name = f"rw_melody_lstm{idx}.mid"
        create_midi_from_list(
            generated_sequence,
            output_path=os.path.join(raw_midi_dir, output_name),
            bpm=88.0,
            channel=0,
            velocity=92,
            note_length_beats=0.5,
        )
        melody_outputs[f"rw_melody_lstm{idx}_seq{selected_seq_idx}"] = output_name

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
    enable_playback=ENABLE_OVERLAY,
)


# Compute and print statistics
# Stats.print_statistics(transition_matrix)
# print_centralities(transition_matrix)

# Plot Degree Distribution Histogram
# degree_distribution = Stats.find_degree_distribution(transition_matrix)
# plot_degree_distribution(degree_distribution)

# TODO: Visualisation: display info about MIDI file, statistics
