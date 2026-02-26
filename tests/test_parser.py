# Test if midi is loaded correctly and transition matrix is generated
import os
import numpy as np
from tonnetz.midi.parser import gen_transition_from_mono_midi


def test_gen_transition_from_mono_midi(filename, channel_number):
    # Path to MIDI file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    midi_file = os.path.join(project_root, "raw_midi", filename)
    transition_matrix = gen_transition_from_mono_midi(midi_file, channel_number)

    # Check if the transition matrix is generated and has the correct shape
    assert transition_matrix is not None, "Transition matrix should not be None"
    assert transition_matrix.shape == (48, 48), "Transition matrix should be 48x48"

    # Check if row sums are either 1 (normalized) or 0 (no transitions)
    row_sums = transition_matrix.sum(axis=1)
    assert np.all((np.isclose(row_sums, 1.0)) | (np.isclose(row_sums, 0.0))), (
        "Each row of the transition matrix should sum to 1 or 0"
    )

    print("Monophonic parser working correctly.")


if __name__ == "__main__":
    filename = "My_Heart_Will_Go_On.mid"
    channel_number = 4
    test_gen_transition_from_mono_midi(filename, channel_number)
