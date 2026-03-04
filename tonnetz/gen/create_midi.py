from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import mido


MIN_NOTE_IDX = 0
MAX_NOTE_IDX = 47
REST_TOKEN = -1
MIDI_NOTE_OFFSET = 36


def create_midi_from_list(
    notes: Sequence[int],
    output_path: str | Path,
    bpm: float = 120.0,
    velocity: int = 64,
    channel: int = 0,
    ticks_per_beat: int = 480,
    note_length_beats: float = 1.0,
    randomize_note_length: bool = False,
    note_length_jitter: float = 0.25,
    random_seed: int | None = None,
) -> Path:
    """
    Create a monophonic MIDI file from Tonnetz note indices.

    Parameters
    ----------
    notes
        Sequence of note indices where values 0..47 map to C2..B5 and -1 is a rest.
    output_path
        Destination `.mid` file path.
    bpm
        Tempo in beats per minute.
    velocity
        MIDI note-on velocity (0..127).
    channel
        MIDI channel (0..15).
    ticks_per_beat
        MIDI resolution.
    note_length_beats
        Base duration of each step in beats. Default is 1.0 (quarter note).
    randomize_note_length
        If True, each step length is multiplied by a random factor.
    note_length_jitter
        Jitter amount for random durations. A value of 0.25 means each step
        is scaled by a factor sampled from [0.75, 1.25].
    random_seed
        Optional seed for deterministic random durations.
    """
    if bpm <= 0:
        raise ValueError("bpm must be > 0")
    if not (0 <= velocity <= 127):
        raise ValueError("velocity must be in range 0..127")
    if not (0 <= channel <= 15):
        raise ValueError("channel must be in range 0..15")
    if ticks_per_beat <= 0:
        raise ValueError("ticks_per_beat must be > 0")
    if note_length_beats <= 0:
        raise ValueError("note_length_beats must be > 0")
    if note_length_jitter < 0:
        raise ValueError("note_length_jitter must be >= 0")

    if randomize_note_length:
        rng = random.Random(random_seed)
    else:
        rng = None

    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    tempo = mido.bpm2tempo(bpm)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

    pending_ticks = 0
    base_ticks = ticks_per_beat * note_length_beats

    for idx, note_idx in enumerate(notes):
        if note_idx == REST_TOKEN:
            pending_ticks += _step_ticks(base_ticks, rng, note_length_jitter)
            continue

        if not (MIN_NOTE_IDX <= note_idx <= MAX_NOTE_IDX):
            raise ValueError(
                f"Invalid note index at position {idx}: {note_idx}. "
                f"Expected {REST_TOKEN} or {MIN_NOTE_IDX}..{MAX_NOTE_IDX}."
            )

        duration_ticks = _step_ticks(base_ticks, rng, note_length_jitter)
        midi_note = note_idx + MIDI_NOTE_OFFSET

        track.append(
            mido.Message(
                "note_on",
                note=midi_note,
                velocity=velocity,
                channel=channel,
                time=pending_ticks,
            )
        )
        track.append(
            mido.Message(
                "note_off",
                note=midi_note,
                velocity=0,
                channel=channel,
                time=duration_ticks,
            )
        )
        pending_ticks = 0

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    mid.save(output)
    return output


def _step_ticks(base_ticks: float,
                jitter: float,) -> int:
    """Return integer step length in ticks, optionally jittered."""
    return max(1, int(round(base_ticks)))
