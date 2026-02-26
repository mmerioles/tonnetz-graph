import mido
import numpy as np


def gen_transition_poly(midi_file: str, target_channel=1) -> np.ndarray:
    """
    Generates a 48x48 Markov transition matrix tracking note transitions from C2 to B5.
    Counts transitions between chords by considering transition between all previous notes to all current notes.

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

    # Ensure the channel number is valid
    if target_channel < 0 or target_channel > 15:
        print("Error: Target channel must be between 0 and 15.")
        return None

    # Track currently active notes as a set of indices
    active_notes = set()
    # Track the previous chord/notes for sequential transitions
    prev_chord = set()

    for msg in mid:
        # Check if message is in the target channel
        if hasattr(msg, "channel") and msg.channel == target_channel:
            if msg.type == "note_on" and msg.velocity > 0:
                # Note being played
                current_note = msg.note

                # Check if the note falls within our 48-note C2-B5 window
                if min_note <= current_note <= max_note:
                    current_note_idx = current_note - min_note

                    # Count transitions from all currently active notes to this new note, and reverse, since its a chord
                    if active_notes:
                        for prev_note_idx in active_notes:
                            transition_matrix[prev_note_idx, current_note_idx] += 1
                            transition_matrix[current_note_idx, prev_note_idx] += 1
                    # Count transition from previous chord to this new note
                    if prev_chord:
                        for prev_note_idx in prev_chord:
                            transition_matrix[prev_note_idx, current_note_idx] += 1

                    # Add this note to the active set
                    active_notes.add(current_note_idx)

            elif msg.type == "note_off" or (
                msg.type == "note_on" and msg.velocity == 0
            ):
                # Note being released
                released_note = msg.note

                # Check if the note falls within our 48-note C2-B5 window
                if min_note <= released_note <= max_note:
                    released_note_idx = released_note - min_note

                    # If prev chord is empty, save the current active notes as the prev chord before any releases
                    if len(prev_chord) == 0 and len(active_notes) > 0:
                        prev_chord = active_notes.copy()

                    # Remove this note from the active set
                    active_notes.discard(released_note_idx)

    # Normalize the transition counts to probabilities
    row_sums = transition_matrix.sum(axis=1, keepdims=True)

    # Avoid division by zero; if a row sum is zero, keep it as zero
    transition_matrix = transition_matrix.astype(float)
    np.divide(transition_matrix, row_sums, out=transition_matrix, where=row_sums != 0)

    # Threshold values below 0.01 to zero for cleaner visualization
    threshold = 0.01
    transition_matrix[transition_matrix < threshold] = 0

    return transition_matrix
