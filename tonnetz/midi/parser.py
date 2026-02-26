import mido
import numpy as np


def gen_transition_from_mono_midi(midi_file: str, target_channel=1) -> np.ndarray:
    """
    Generates a 48x48 Markov transition matrix tracking note transitions from C2 to B5.

    Parameters
    ----------
    midi_file : str
        Path to the MIDI file.
    target_channel : int
        The MIDI channel number to parse (defaults to 1).

    Returns
    -------
        np.ndarray: A 48x48 Markov transition matrix.
    """

    # Initializing the matrix
    num_notes = 48
    min_note = 36  # MIDI note number for C2
    max_note = min_note + num_notes - 1  # MIDI for B5

    transition_matrix = np.zeros((num_notes, num_notes), dtype=int)

    try:
        mid = mido.MidiFile(midi_file)
    except Exception as e:
        print(f"Error loading MIDI file: {e}")
        return None

    # Ensure the channel number is standard (0 through 15)
    if target_channel < 0 or target_channel > 15:
        print("Error: Target channel must be between 0 and 15.")
        return None

    prev_note_idx = None

    for msg in mid:
        # Check if message is in the target channel
        if hasattr(msg, "channel") and msg.channel == target_channel:
            # Note being played => note_on with velocity > 0
            if msg.type == "note_on" and msg.velocity > 0:
                current_note = msg.note

                # Check if the note falls within our 48-note C2-B5 window
                if min_note <= current_note <= max_note:
                    current_note_idx = current_note - min_note

                    # If we have a valid previous note, record the transition
                    if prev_note_idx is not None:
                        transition_matrix[prev_note_idx, current_note_idx] += 1

                    prev_note_idx = current_note_idx
                else:
                    # Reset prev note
                    prev_note_idx = None

    # Normalize the transition counts to probabilities
    row_sums = transition_matrix.sum(axis=1, keepdims=True)

    # Avoid division by zero; if a row sum is zero, keep it as zero
    transition_matrix = transition_matrix.astype(float)
    np.divide(transition_matrix, row_sums, out=transition_matrix, where=row_sums != 0)

    # Threshold values below 0.01 to zero for cleaner visualization
    threshold = 0.01
    transition_matrix[transition_matrix < threshold] = 0

    return transition_matrix
