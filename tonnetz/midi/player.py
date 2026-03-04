from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import time
import sys
import mido


MIN_NOTE = 36
MAX_NOTE = 83  # inclusive


@dataclass(frozen=True)
class MidiEvent:
    t: float       # seconds since start
    kind: str      # "on" or "off"
    note: int      # MIDI note number (0..127)
    vel: int       # velocity (0-127)


def _is_windows() -> bool:
    return sys.platform.startswith("win")


def _is_macos() -> bool:
    return sys.platform.startswith("darwin")


def get_initial_bpm(midi_file: str) -> float:
    """
    Return the first tempo found in the file as BPM.
    Falls back to 120.0 if there is no tempo meta.
    """
    mid = mido.MidiFile(midi_file)
    tempo_us = 500_000  # default 120 BPM
    for msg in mido.merge_tracks(mid.tracks):
        if msg.type == "set_tempo":
            tempo_us = msg.tempo
            break
    return mido.tempo2bpm(tempo_us)


def midi_to_events_ticks(
    midi_file: str,
    target_channel: Optional[int] = 0,
    exclude_drums: bool = True,
) -> List[MidiEvent]:
    """
    Convert a MIDI file into a flat, time-sorted list of MidiEvent objects.
    Event times are expressed in SECONDS and respect all tempo changes.

    Parameters
    ----------
    midi_file:
        Path to the MIDI file.
    target_channel:
        If an integer (0-15), only events from that channel are used.
        If None, events from all non-drum channels are used.
    """
    mid = mido.MidiFile(midi_file)
    ticks_per_beat = mid.ticks_per_beat
    tempo = 500_000  # default 120 BPM in microseconds

    events: List[MidiEvent] = []
    abs_sec = 0.0

    for msg in mido.merge_tracks(mid.tracks):
        if msg.time:
            abs_sec += mido.tick2second(msg.time, ticks_per_beat, tempo)

        if msg.type == "set_tempo":
            tempo = msg.tempo
            continue

        channel = getattr(msg, "channel", None)
        if channel is None:
            continue

        if target_channel is not None and channel != target_channel:
            continue

        if msg.type == "note_on" and msg.velocity > 0:
            events.append(MidiEvent(abs_sec, "on", msg.note, int(msg.velocity)))
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            # Preserve note_off events without velocity (velocity 0 is semantically correct)
            events.append(MidiEvent(abs_sec, "off", msg.note, 0))
    return events


def scale_events_bpm(
    events: List[MidiEvent],
    original_bpm: float,
    new_bpm: float,
) -> List[MidiEvent]:
    """
    Return a new event list whose times are scaled so that playback
    at new_bpm keeps the same beat positions as original_bpm.

    This assumes `events` were generated at original_bpm; for
    constant-tempo pieces this is exactly what we want. For files
    with many tempo changes it is still a reasonable global speed
    adjustment.
    """
    if new_bpm <= 0:
        return events
    scale = float(original_bpm) / float(new_bpm)
    return [MidiEvent(e.t * scale, e.kind, e.note, e.vel) for e in events]


class FluidSynthPlayer:
    def __init__(self, soundfont_path: str, gain: float = 0.6):
        try:
            import fluidsynth
        except ImportError:
            raise ImportError("fluidsynth not installed. Install with: pip install fluidsynth")

        # Base synth. We keep the default sample rate but tighten the
        # audio buffer config before starting to reduce end-to-end
        # latency (less "drag" between note scheduling and playback).
        self.fs = fluidsynth.Synth(gain=gain)

        try:
            self.fs.setting("audio.period-size", 1024)
            self.fs.setting("audio.periods", 8)
            self.fs.setting("synth.reverb.active", 0)
            self.fs.setting("synth.chorus.active", 0)
        except Exception:
            # If settings are unsupported, just fall back to defaults.
            pass

        # Cross-platform driver selection
        driver = self._select_driver()
        try:
            self.fs.start(driver=driver)
        except Exception as e:
            print(f"Warning: Failed to start with driver '{driver}': {e}")
            print("Attempting to start without explicit driver...")
            self.fs.start()

        # Load soundfont
        try:
            self.sfid = self.fs.sfload(soundfont_path)
        except Exception as e:
            self.fs.delete()
            raise RuntimeError(f"Failed to load soundfont '{soundfont_path}': {e}")

        # Select acoustic grand piano on channel 0
        self.fs.program_select(0, self.sfid, 0, 0)
        self._active_notes = set()

    def _select_driver(self) -> str:
        """Select appropriate audio driver for the current platform."""
        if _is_windows():
            return "dsound"  # DirectSound on Windows
        elif _is_macos():
            return "coreaudio"  # CoreAudio on macOS
        else:
            return "pulseaudio"  # PulseAudio on Linux (fallback: alsa)

    def note_on(self, midi_note: int, velocity: int):
        """Start playing a note."""
        # Allow the full MIDI range for audio; visualization code is responsible
        # for mapping/limiting to the Tonnetz node range.
        if not (0 <= midi_note <= 127):
            return
        self.fs.noteon(0, midi_note, int(velocity))
        self._active_notes.add(midi_note)

    def note_off(self, midi_note: int):
        """Stop playing a note."""
        if not (0 <= midi_note <= 127):
            return
        self.fs.noteoff(0, midi_note)
        self._active_notes.discard(midi_note)

    def flush(self):
        """Flush audio buffer to prevent crackling and ensure notes are played."""
        try:
            # This forces the synthesizer to process pending events
            self.fs.get_cpu_load()  # Minimal operation that triggers internal processing
        except:
            pass

    def all_notes_off(self):
        """Stop all currently playing notes."""
        notes_copy = list(self._active_notes)
        for note in notes_copy:
            self.note_off(note)
        self.flush()

    def close(self):
        """Clean up and close the synthesizer."""
        try:
            self.all_notes_off()
            time.sleep(0.1)  # Allow final notes to decay
            self.fs.delete()
        except Exception as e:
            print(f"Warning during cleanup: {e}")


def play_midi_file(
    midi_file: str,
    soundfont_path: str,
    target_channel: Optional[int] = None,
    bpm_override: Optional[float] = None,
):
    """
    Play a MIDI file using FluidSynth.
    
    Parameters:
    -----------
    midi_file : str
        Path to the MIDI file
    soundfont_path : str
        Path to the SoundFont (.sf2) file
    target_channel : int | None
        If an integer (0–15), only that channel is played.
        If None, all non-drum channels are mixed together.
    bpm_override : float, optional
        Override the tempo (BPM). If None, uses original tempo.
    """
    # Extract events in SECONDS using the MIDI's tempo map.
    events = midi_to_events_ticks(midi_file, target_channel=target_channel)
    if not events:
        if target_channel is None:
            print("No note events found in the MIDI file.")
        else:
            print(f"No events found in channel {target_channel}")
        return

    # Optionally apply a global BPM override by scaling times.
    base_bpm = get_initial_bpm(midi_file)
    effective_bpm = base_bpm
    if bpm_override and bpm_override > 0:
        events = scale_events_bpm(events, original_bpm=base_bpm, new_bpm=bpm_override)
        effective_bpm = float(bpm_override)

    # Create player
    player = FluidSynthPlayer(soundfont_path)
    
    try:
        print(f"Playing {len(events)} events at {effective_bpm:.2f} BPM...")
        start_time = time.perf_counter()
        event_idx = 0

        while event_idx < len(events):
            current_time = time.perf_counter() - start_time
            
            # Process all events that should have occurred by now
            while event_idx < len(events) and events[event_idx].t <= current_time:
                event = events[event_idx]
                if event.kind == "on":
                    player.note_on(event.note, event.vel)
                elif event.kind == "off":
                    player.note_off(event.note)
                event_idx += 1
            
            # Flush periodically to prevent crackling
            if event_idx % 10 == 0:
                player.flush()
            
            # Sleep only until (just before) the next event to avoid
            # accumulating lag while still preventing busy-waiting.
            if event_idx < len(events):
                next_t = events[event_idx].t
                remaining = next_t - current_time
                if remaining > 0.003:
                    # Wake up a little early (max 5ms) so we can
                    # dispatch clusters of near-simultaneous events
                    # without quantising everything to a coarse grid.
                    time.sleep(min(remaining, 0.005))

        # Wait for final notes to finish
        final_event_time = events[-1].t if events else 0
        while time.perf_counter() - start_time < final_event_time + 2.0:
            time.sleep(0.01)

    finally:
        player.close()
        print("Playback complete.")