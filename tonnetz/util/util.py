def create_note_labels() -> dict:
    """Create a mapping from node indices to note names (0=C2, 47=B5)."""
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return {i: f"{note_names[i % 12]}{i // 12 + 2}" for i in range(48)}
