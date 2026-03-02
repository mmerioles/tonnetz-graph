"""
Convenience CLI script to test MIDI playback using the shared FluidSynth
pipeline used by the Tonnetz visualization.

Examples (run from repo root):
    uv run python -m scripts.play
    uv run python -m scripts.play --midi beethooven-3rd-movement.mid
    uv run python -m scripts.play --midi path/to/file.mid --bpm 90
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from tonnetz.midi.player import play_midi_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a MIDI file with FluidSynth.")
    parser.add_argument(
        "--midi",
        type=str,
        default="beethooven-3rd-movement.mid",
        help="MIDI filename (looked up in raw_midi/) or absolute path.",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=None,
        help="MIDI channel to play (0–15). Omit to use all non-drum channels.",
    )
    parser.add_argument(
        "--bpm",
        type=float,
        default=None,
        help="Override tempo (BPM). If omitted, use the original tempo.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    midi_arg = args.midi

    # Resolve MIDI path
    if os.path.isabs(midi_arg):
        midi_path = Path(midi_arg)
    else:
        midi_path = project_root / "raw_midi" / midi_arg

    if not midi_path.exists():
        print(f"Error: MIDI file not found at '{midi_path}'.")
        return

    soundfont_path = project_root / "raw_midi" / "GeneralUser-GS.sf2"
    if not soundfont_path.exists():
        print(f"Error: SoundFont not found at '{soundfont_path}'.")
        return

    play_midi_file(
        midi_file=str(midi_path),
        soundfont_path=str(soundfont_path),
        target_channel=args.channel,
        bpm_override=args.bpm,
    )


if __name__ == "__main__":
    main()