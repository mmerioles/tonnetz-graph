import argparse
import os
from pathlib import Path
from tonnetz.midi.player import play_midi_file

DEFAULT_FILE="My_Heart_Will_Go_On_combined.mid"

def main() -> None:
    parser = argparse.ArgumentParser(description="Play a MIDI file with FluidSynth.")
    parser.add_argument(
        "--midi",
        type=str,
        default=DEFAULT_FILE,
        help="MIDI filename (looked up in raw_midi/) or absolute path.",
    )
    parser.add_argument(
        "--bpm",
        type=float,
        default=None,
        help="Override tempo (BPM). If omitted, use the original tempo.",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=None,
        help="If set (0-15), only play this MIDI channel. Default plays all channels.",
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
