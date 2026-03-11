from __future__ import annotations

import ast
import argparse
import random
import shutil
import subprocess
from pathlib import Path

import mido


CSV_FILE = "lstm_generated_seq (6).csv"
NUM_SEQUENCES = 12
INCLUDE_SEQUENCES = [222, 333, 444]
START_NOTE = "E4"
BPM = 120
NOTE_RESOLUTION = "16th"
#NOTE_RESOLUTION = "8th"


NOTE_TO_SEMITONE = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}

RANDOM_SEED = 7


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    csv_path = root / "data" / CSV_FILE
    soundfont_path = root / "raw_midi" / "GeneralUser-GS.sf2"
    output_dir = root / "wav"
    output_dir.mkdir(exist_ok=True)

    sequences = load_sequences(csv_path)
    start_note = note_to_midi(START_NOTE)
    selected_indices = choose_sequences(len(sequences))

    for index in selected_indices:
        tokens = sequences[index]
        midi_path = output_dir / f"{csv_path.stem}_seq_{index + 1:03d}.mid"
        wav_path = output_dir / f"{csv_path.stem}_seq_{index + 1:03d}.wav"

        write_midi(tokens, start_note, midi_path, NOTE_RESOLUTION)
        render_wav(midi_path, wav_path, soundfont_path)
        midi_path.unlink(missing_ok=True)
        print(f"Wrote {wav_path}")


def load_sequences(csv_path: Path) -> list[list[int]]:
    sequences = []
    for line in csv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line[0] == line[-1] == "'":
            line = line[1:-1]
        sequences.append(ast.literal_eval(line))
    return sequences


def choose_sequences(total_sequences: int) -> list[int]:
    selected = []

    for sequence in INCLUDE_SEQUENCES:
        index = sequence - 1
        if 0 <= index < total_sequences and index not in selected:
            selected.append(index)

    remaining = max(0, NUM_SEQUENCES - len(selected))
    pool = [index for index in range(total_sequences) if index not in selected]
    selected.extend(random.Random(RANDOM_SEED).sample(pool, remaining))
    return selected


def note_to_midi(note: str) -> int:
    name = note[:-1]
    octave = int(note[-1])
    return NOTE_TO_SEMITONE[name] + (octave + 1) * 12


def token_to_semitones(token: int) -> int:
    # return token - 25 if token <= 24 else token - 24
    return token - 26


def write_midi(
    tokens: list[int],
    start_note: int,
    output_path: Path,
    note_resolution: str,
) -> None:
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(BPM), time=0))

    current_note = start_note
    active_note: int | None = None
    active_steps = 0
    rest_steps = 0
    step_ticks = 240 if note_resolution == "8th" else 120

    def flush_note() -> None:
        nonlocal active_note, active_steps, rest_steps
        if active_note is None:
            return
        track.append(
            mido.Message(
                "note_on",
                note=active_note,
                velocity=72,
                channel=0,
                time=rest_steps * step_ticks,
            )
        )
        track.append(
            mido.Message(
                "note_off",
                note=active_note,
                velocity=0,
                channel=0,
                time=active_steps * step_ticks,
            )
        )
        active_note = None
        active_steps = 0
        rest_steps = 0

    for token in tokens:
        if token == 0:
            flush_note()
            rest_steps += 1
        elif token == 1:
            if active_note is None:
                active_note = current_note
                active_steps = 1
            else:
                active_steps += 1
        else:
            flush_note()
            current_note += token_to_semitones(token)
            active_note = current_note
            active_steps = 1

    flush_note()
    midi.save(output_path)


def render_wav(midi_path: Path, wav_path: Path, soundfont_path: Path) -> None:
    fluidsynth = shutil.which("fluidsynth")
    subprocess.run(
        [
            fluidsynth,
            "-ni",
            "-T",
            "wav",
            "-F",
            str(wav_path),
            str(soundfont_path),
            str(midi_path),
        ],
        check=True,
        capture_output=True,
    )


if __name__ == "__main__":
    main()
