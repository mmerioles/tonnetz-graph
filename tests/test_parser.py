# Test if midi is loaded correctly and transition matrix is generated
import os
import numpy as np
from tonnetz.midi.parser import gen_transition_from_mono_midi

filename = "My_Heart_Will_Go_On.mid"
channel_number = 4


def test_gen_transition_from_mono_midi():
    # Path to MIDI file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    midi_file = os.path.join(project_root, "raw_midi", filename)
    transition_matrix = gen_transition_from_mono_midi(midi_file, channel_number)

    # Generate the transition matrix
    transition_matrix = gen_transition_from_mono_midi(midi_file)

    # Check if the transition matrix is generated and has the correct shape
    assert transition_matrix is not None, "Transition matrix should not be None"
    assert transition_matrix.shape == (48, 48), "Transition matrix should be 48x48"

    # Check if row sums are 1
    row_sums = transition_matrix.sum(axis=1)
    assert np.allclose(row_sums, 1), "Each row of the transition matrix should sum to 1"

    print("Monophonic parser working correctly.")
