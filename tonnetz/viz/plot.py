import os
import time
import threading
import csv
import re
from pathlib import Path

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import mido
from matplotlib.widgets import RadioButtons, Button, TextBox
try:
    from tonnetz.midi.player import (
        midi_to_events_ticks,
        FluidSynthPlayer,
        MidiEvent,
        get_initial_bpm,
        scale_events_bpm,
    )
    _AUDIO_AVAILABLE = True
except ImportError as _e:  
    _AUDIO_AVAILABLE = False
    _AUDIO_IMPORT_ERROR = _e
from tonnetz.util.util import create_note_labels

MIN_NOTE = 36
MAX_NOTE = 83

_MODULE_DIR = Path(__file__).resolve().parent         
_PROJECT_ROOT = _MODULE_DIR.parents[2]                 
_MIDI_DIR = _PROJECT_ROOT / "tonnetz-graph" / "raw_midi"
_DEFAULT_SOUNDFONT = _MIDI_DIR / "GeneralUser-GS.sf2"
_CHANNEL_MAP_CSV = _MIDI_DIR / "mono_channels.csv"


def _parse_channel_list(raw: str | None) -> list[int]:
    if raw is None:
        return []
    text = str(raw).strip()
    if not text:
        return []

    out: list[int] = []
    for token in re.split(r"[|;/\s]+", text):
        tok = token.strip()
        if not tok:
            continue
        if not tok.lstrip("-").isdigit():
            continue
        ch = int(tok)
        if 0 <= ch <= 15:
            out.append(ch)
    return sorted(set(out))


def _normalize_file_key(path_like: str) -> str:
    return Path(path_like).name.lower().strip()


def _resolve_role_config(
    role_map: dict[str, dict[str, list[int]]],
    midi_path: str,
) -> dict[str, list[int]]:
    """
    Resolve channel-role config for a MIDI path with a fallback for combined files.
    Example fallback: song_combined.mid -> song.mid
    """
    key = _normalize_file_key(midi_path)
    cfg = role_map.get(key)
    if cfg is not None:
        return cfg

    stem = Path(key).stem
    suffix = Path(key).suffix
    if stem.endswith("_combined"):
        base_key = f"{stem[:-9]}{suffix}"
        cfg = role_map.get(base_key)
        if cfg is not None:
            return cfg

    return {}


def _load_role_channels(csv_path: Path) -> dict[str, dict[str, list[int]]]:
    """
    Load channel role mapping from CSV.

    Supported rows:
    - filename, channel                (legacy: melody only)
    - filename, melody_channel, chords_channel
    - with headers using fields like: file/filename/midi, melody, chord/chords
    """
    mapping: dict[str, dict[str, list[int]]] = {}
    if not csv_path.exists():
        return mapping

    lines: list[str] = []
    with csv_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                lines.append(s)
    if not lines:
        return mapping

    has_header = any(
        key in lines[0].lower()
        for key in ("file", "filename", "midi", "melody", "chord")
    )

    if has_header:
        reader = csv.DictReader(lines)
        for row in reader:
            if not row:
                continue
            filename = (
                row.get("filename")
                or row.get("file")
                or row.get("midi")
                or row.get("track")
            )
            if not filename:
                continue
            melody = (
                row.get("melody")
                or row.get("melody_channel")
                or row.get("melody_channels")
                or row.get("lead")
            )
            chords = (
                row.get("chords")
                or row.get("chord")
                or row.get("chord_channel")
                or row.get("chord_channels")
                or row.get("harmony")
            )
            key = _normalize_file_key(filename)
            mapping[key] = {
                "melody": _parse_channel_list(melody),
                "chords": _parse_channel_list(chords),
            }
    else:
        reader = csv.reader(lines)
        for row in reader:
            if not row:
                continue
            filename = str(row[0]).strip()
            if not filename:
                continue
            key = _normalize_file_key(filename)
            melody_raw = row[1] if len(row) > 1 else ""
            chords_raw = row[2] if len(row) > 2 else ""
            mapping[key] = {
                "melody": _parse_channel_list(melody_raw),
                "chords": _parse_channel_list(chords_raw),
            }

    return mapping


def _events_for_channels(midi_path: str, channels: list[int] | None) -> list[MidiEvent]:
    if not channels:
        return []

    merged: list[MidiEvent] = []
    for ch in sorted(set(channels)):
        merged.extend(midi_to_events_ticks(midi_path, target_channel=ch))
    merged.sort(key=lambda e: e.t)
    return merged


def _build_tempo_segments(mid: mido.MidiFile) -> list[tuple[int, float, int]]:
    """
    Build tempo segments as tuples of:
    (start_tick, start_sec, tempo_us_per_beat)
    """
    ticks_per_beat = mid.ticks_per_beat
    tempo = 500_000  # default 120 BPM
    abs_tick = 0
    abs_sec = 0.0
    segments: list[tuple[int, float, int]] = [(0, 0.0, tempo)]

    for msg in mido.merge_tracks(mid.tracks):
        if msg.time:
            abs_tick += int(msg.time)
            abs_sec += mido.tick2second(msg.time, ticks_per_beat, tempo)
        if msg.type == "set_tempo":
            tempo = int(msg.tempo)
            segments.append((abs_tick, abs_sec, tempo))

    return segments


def _tick_to_sec(
    tick: int,
    segments: list[tuple[int, float, int]],
    ticks_per_beat: int,
) -> float:
    # Linear scan is fine for small tempo maps.
    seg_i = 0
    for i in range(1, len(segments)):
        if segments[i][0] > tick:
            break
        seg_i = i
    start_tick, start_sec, tempo = segments[seg_i]
    delta = tick - start_tick
    return start_sec + mido.tick2second(delta, ticks_per_beat, tempo)


def _events_for_tracks(midi_path: str, track_indices: list[int] | None) -> list[MidiEvent]:
    if not track_indices:
        return []

    mid = mido.MidiFile(midi_path)
    ticks_per_beat = mid.ticks_per_beat
    segments = _build_tempo_segments(mid)
    merged: list[MidiEvent] = []

    note_track_indices = [
        i
        for i, track in enumerate(mid.tracks)
        if any(
            (msg.type in ("note_on", "note_off")) and getattr(msg, "note", None) is not None
            for msg in track
        )
    ]

    # Interpret provided indices as ordinal note-track positions first:
    # 0 => first note-bearing track, 1 => second note-bearing track, etc.
    resolved_indices: list[int] = []
    unique_requested = sorted(set(track_indices))
    if note_track_indices and all(0 <= idx < len(note_track_indices) for idx in unique_requested):
        resolved_indices = [note_track_indices[idx] for idx in unique_requested]
    else:
        # Fallback: treat as absolute MIDI track indices.
        resolved_indices = unique_requested

    for track_i in resolved_indices:
        if track_i < 0 or track_i >= len(mid.tracks):
            continue
        abs_tick = 0
        for msg in mid.tracks[track_i]:
            abs_tick += int(msg.time)
            note = getattr(msg, "note", None)
            if note is None:
                continue

            t_sec = _tick_to_sec(abs_tick, segments, ticks_per_beat)
            if msg.type == "note_on" and msg.velocity > 0:
                merged.append(MidiEvent(t_sec, "on", int(note), int(msg.velocity)))
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                merged.append(MidiEvent(t_sec, "off", int(note), 0))

    merged.sort(key=lambda e: e.t)
    return merged

def plot_graph(
    input_graph: nx.DiGraph,
    show_isolated_nodes: bool = False,
    show: bool = True,
    name: str = "Network Graph",
    centralities: dict | None = None,
    overlay_chord_midi_name: str | None = None,
    overlay_melody_options: dict[str, str] | None = None,
):
    G = input_graph.copy()

    if not show_isolated_nodes:
        isolated = list(nx.isolates(G))
        G.remove_nodes_from(isolated)

    # Labels only for nodes that exist in G
    all_note_labels = create_note_labels()
    note_labels = {n: all_note_labels[n] for n in G.nodes()}

    # Layout + styling
    node_pos = nx.kamada_kawai_layout(G, scale=5)
    degree_centrality = dict(G.in_degree())
    node_colors = [degree_centrality[n] for n in G.nodes()]
    node_sizes = [max(degree_centrality[n] * 60, 40) for n in G.nodes()]
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    edge_widths = [w * 0.3 for w in edge_weights]

    # Create figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    # Draw graph edges
    nx.draw_networkx_edges(
        G,
        node_pos,
        ax=ax,
        arrows=True,
        connectionstyle="arc3,rad=0",
        arrowsize=3,
        width=edge_widths,
        edge_color="black",
    )

    # Draw nodes
    node_artist = nx.draw_networkx_nodes(
        G,
        node_pos,
        ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.Blues,
        alpha=0.95,
    )

    # Draw labels
    nx.draw_networkx_labels(
        G,
        node_pos,
        labels=note_labels,
        ax=ax,
        font_color="black",
        font_family="Arial",
        font_size=12,
    )

    plt.title(
        name,
        fontsize=18,
        fontweight="bold",
        fontfamily="Times New Roman",
        color="black",
        pad=20,
    )

    overlay = None

    if _AUDIO_AVAILABLE:
        chord_midi_path = overlay_chord_midi_name or name
        if not os.path.isabs(chord_midi_path):
            candidate = _MIDI_DIR / chord_midi_path
            if candidate.exists():
                chord_midi_path = str(candidate)

        resolved_melody_options: dict[str, str] = {}
        if overlay_melody_options:
            for label, path_like in overlay_melody_options.items():
                path = path_like
                if not os.path.isabs(path):
                    candidate = _MIDI_DIR / path
                    if candidate.exists():
                        path = str(candidate)
                resolved_melody_options[label] = path

        soundfont_path = str(_DEFAULT_SOUNDFONT) if _DEFAULT_SOUNDFONT.exists() else None

        if os.path.exists(chord_midi_path):
            overlay = TonnetzRealtimeOverlay(
                fig=fig,
                ax=ax,
                node_artist=node_artist,
                nodes_in_graph=list(G.nodes()),
                node_pos=node_pos,
                chord_midi_path=chord_midi_path,
                soundfont_path=soundfont_path,
                melody_midi_options=resolved_melody_options,
                melody_track=0,
                chord_track=1,
            )
    else:
        print(
            "Audio overlay disabled (missing audio dependencies such as 'mido' or 'pyfluidsynth'):",
            _AUDIO_IMPORT_ERROR,
        )

    # --- Centrality overlay (existing logic) ---
    if centralities:
        nodes = list(G.nodes())

        deg_norm = centralities["deg"]
        btw_str = centralities["btw"]
        eig_str = centralities["eig"]

        deg_vals = np.array([deg_norm.get(int(n), 0.0) for n in nodes], dtype=float)
        deg_min = float(deg_vals.min()) if deg_vals.size else 0.0
        deg_max = float(deg_vals.max()) if deg_vals.size else 1.0
        if deg_max == deg_min:
            deg_max = deg_min + 1.0

        btw = {int(k): float(v) for k, v in btw_str.items()}
        eig = {int(k): float(v) for k, v in eig_str.items()}

        def rescale_to_degree_range(vals: np.ndarray) -> np.ndarray:
            if vals.size == 0:
                return vals
            vmin = float(vals.min())
            vmax = float(vals.max())
            if vmax == vmin:
                return np.full(vals.shape, deg_min, dtype=float)
            t = (vals - vmin) / (vmax - vmin)
            return deg_min + t * (deg_max - deg_min)

        def metric_values(metric: str) -> np.ndarray:
            if metric == "degree":
                base = np.array([deg_norm.get(int(n), 0.0) for n in nodes], dtype=float)
                return rescale_to_degree_range(base)
            if metric == "betweenness":
                base = np.array([btw.get(int(n), 0.0) for n in nodes], dtype=float)
                return rescale_to_degree_range(base)
            if metric == "eigenvector":
                base = np.array([eig.get(int(n), 0.0) for n in nodes], dtype=float)
                return rescale_to_degree_range(base)
            raise ValueError(metric)

        def sizes_from(vals: np.ndarray) -> np.ndarray:
            return np.array([max(v * 6000, 40) for v in vals], dtype=float)

        # Add radio button controls
        rax = fig.add_axes([0.02, 0.35, 0.17, 0.25])
        radio = RadioButtons(rax, ("degree", "betweenness", "eigenvector"), active=0)
        rax.set_title("Centrality", fontsize=11)

        def on_change(label: str):
            vals = metric_values(label)
            node_artist.set_array(vals)
            node_artist.set_sizes(sizes_from(vals))
            node_artist.set_clim(deg_min, deg_max)
            fig.canvas.draw_idle()

        radio.on_clicked(on_change)
        on_change("degree")

    # Clean up on window close
    def on_close(event):
        if overlay:
            overlay.close()
    
    fig.canvas.mpl_connect('close_event', on_close)

    if show:
        plt.show()

    return fig, overlay


def plot_degree_distribution(
    degree_distribution: dict[int, float],
    show: bool = True
) -> None:
    """Plot histogram of degree distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))
    degrees = list(degree_distribution.keys())
    counts = list(degree_distribution.values())
    ax.bar(degrees, counts, color="blue", alpha=0.7)
    ax.set_xlabel("Degree")
    ax.set_ylabel("Count")
    ax.set_title("Degree Distribution Histogram")
    plt.tight_layout()
    if show:
        plt.show()


class TonnetzRealtimeOverlay:
    def __init__(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        node_artist,
        nodes_in_graph: list[int],
        node_pos: dict[int, np.ndarray],
        chord_midi_path: str,
        soundfont_path: str | None = None,
        melody_midi_options: dict[str, str] | None = None,
        melody_track: int = 0,
        chord_track: int = 1,
    ):
        self.fig = fig
        self.ax = ax
        self.node_artist = node_artist

        self.nodes = list(nodes_in_graph)
        self.node_to_i = {n: i for i, n in enumerate(self.nodes)}

        self.node_pos = node_pos
        self.xy = np.array([self.node_pos[n] for n in self.nodes], dtype=float)

        self.background = None
        self._full_bg = None
        self._blit_ready = False
        self._needs_full_redraw = False
        self.fig.canvas.mpl_connect("draw_event", self._on_draw)

        sizes = np.array(self.node_artist.get_sizes(), dtype=float)
        if sizes.size == 1 and len(self.nodes) > 1:
            sizes = np.repeat(sizes, repeats=len(self.nodes))
        self.base_sizes = sizes.copy()

        self.melody_artist = ax.scatter(
            [],
            [],
            s=[],
            facecolors="none",
            edgecolors="red",
            linewidths=2.5,
            zorder=10,
            animated=True,
        )
        self.chord_artist = ax.scatter(
            [],
            [],
            s=[],
            facecolors="none",
            edgecolors="limegreen",
            linewidths=2.2,
            zorder=9,
            animated=True,
        )

        self.active_melody_nodes: set[int] = set()
        self.active_chord_nodes: set[int] = set()
        self.active_melody_note_counts: dict[int, int] = {}
        self.active_chord_note_counts: dict[int, int] = {}

        self.melody_track = melody_track
        self.chord_track = chord_track
        self.chord_midi_path = chord_midi_path
        self.melody_midi_options = melody_midi_options or {}
        self.melody_labels = list(self.melody_midi_options.keys())
        self.selected_melody_label = self.melody_labels[0] if self.melody_labels else None

        self.chord_events0 = _events_for_tracks(self.chord_midi_path, [self.chord_track])
        self.melody_events0 = self._load_selected_melody_events()
        self.events0: list[tuple[float, str, MidiEvent]] = [
            (e.t, "melody", e) for e in self.melody_events0
        ] + [
            (e.t, "chords", e) for e in self.chord_events0
        ]
        self.events0.sort(key=lambda x: x[0])
        self.base_bpm = float(get_initial_bpm(self.chord_midi_path))
        self.events: list[tuple[float, str, MidiEvent]] = self.events0[:]

        self.audio = None
        self._audio_channel_by_role = {"melody": 0, "chords": 1}
        self._audio_lock = threading.Lock()
        if soundfont_path:
            self.audio = FluidSynthPlayer(soundfont_path)
            self.audio.setup_channel(0, bank=0, preset=0, volume=120, pan=64)
            self.audio.setup_channel(1, bank=0, preset=0, volume=102, pan=64)

        self.is_playing = False
        self.t0 = 0.0

        ax_play = fig.add_axes([0.85, 0.92, 0.12, 0.06])
        self.btn = Button(ax_play, "Play")

        ax_bpm = fig.add_axes([0.70, 0.92, 0.13, 0.06])
        self.bpm_box = TextBox(ax_bpm, "BPM", initial=f"{self.base_bpm:.2f}")
        self.melody_radio = None
        if self.melody_labels:
            # Keep selector comfortably inside figure bounds with enough room
            # for filename-style labels.
            rax_melody = fig.add_axes([0.72, 0.62, 0.26, 0.28])
            self.melody_radio = RadioButtons(rax_melody, tuple(self.melody_labels), active=0)
            rax_melody.set_title("Melody", fontsize=10)
            for lbl in self.melody_radio.labels:
                lbl.set_fontsize(8)
                lbl.set_clip_on(True)
            self.melody_radio.on_clicked(self._on_change_melody)

        self.btn.on_clicked(self._toggle_play)

        self._last_draw = 0.0
        self.target_fps = self._detect_target_fps()
        interval_ms = max(1000 // max(self.target_fps, 1), 1)
        self.render_timer = fig.canvas.new_timer(interval=interval_ms)
        self.render_timer.add_callback(self._on_render_tick)

        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._playback_thread: threading.Thread | None = None
        self._playback_done = False
        self._dirty_visual = False

        # Force an initial full render after controls are created so widget
        # axes are visible immediately (not only after a resize redraw).
        self.fig.canvas.draw()

    def _detect_target_fps(self) -> int:
        raw = os.getenv("TONNETZ_OVERLAY_FPS", "").strip()
        if raw and raw.isdigit():
            val = int(raw)
            if 5 <= val <= 120:
                return val
        # Lower default FPS for broader low-spec compatibility.
        return 30

    def _load_selected_melody_events(self) -> list[MidiEvent]:
        if not self.selected_melody_label:
            return []
        path = self.melody_midi_options.get(self.selected_melody_label)
        if not path or not os.path.exists(path):
            return []
        return _events_for_tracks(path, [self.melody_track])

    def _rebuild_events_for_selection(self):
        self.melody_events0 = self._load_selected_melody_events()
        self.events0 = [(e.t, "melody", e) for e in self.melody_events0] + [
            (e.t, "chords", e) for e in self.chord_events0
        ]
        self.events0.sort(key=lambda x: x[0])
        self.events = self.events0[:]

    def _on_change_melody(self, label: str):
        self.selected_melody_label = label
        was_playing = self.is_playing
        self.stop()
        self._rebuild_events_for_selection()
        if was_playing:
            self.start()

    def _toggle_play(self, _event):
        if not self.is_playing:
            self.start()
        else:
            self.stop()

    def start(self):
        self.stop()

        self.events = self.events0[:]

        bpm = self._read_bpm()
        if bpm is not None and bpm > 0 and abs(bpm - self.base_bpm) > 1e-6:
            melody_scaled = scale_events_bpm(
                self.melody_events0,
                original_bpm=self.base_bpm,
                new_bpm=bpm,
            )
            chord_scaled = scale_events_bpm(
                self.chord_events0,
                original_bpm=self.base_bpm,
                new_bpm=bpm,
            )
            self.events = [(e.t, "melody", e) for e in melody_scaled] + [
                (e.t, "chords", e) for e in chord_scaled
            ]
            self.events.sort(key=lambda x: x[0])

        self.is_playing = True
        self.t0 = time.perf_counter()

        with self._state_lock:
            self.active_melody_nodes.clear()
            self.active_chord_nodes.clear()
            self.active_melody_note_counts.clear()
            self.active_chord_note_counts.clear()
            self._dirty_visual = True
            self._playback_done = False
            self._needs_full_redraw = True
        self._apply_highlight()

        self.btn.label.set_text("Stop")
        self._stop_event.clear()
        self._playback_thread = threading.Thread(
            target=self._playback_loop,
            args=(self.events, self.t0),
            daemon=True,
        )
        self._playback_thread.start()
        self.render_timer.start()

    def stop(self):
        self._stop_event.set()
        if (
            self._playback_thread
            and self._playback_thread.is_alive()
            and threading.current_thread() is not self._playback_thread
        ):
            self._playback_thread.join(timeout=0.25)

        self.is_playing = False
        self.render_timer.stop()
        self.btn.label.set_text("Play")

        with self._state_lock:
            melody_snapshot = list(self.active_melody_note_counts.keys())
            chord_snapshot = list(self.active_chord_note_counts.keys())

        if self.audio:
            with self._audio_lock:
                for midi_note in melody_snapshot:
                    self.audio.note_off(midi_note, channel=self._audio_channel_by_role["melody"])
                for midi_note in chord_snapshot:
                    self.audio.note_off(midi_note, channel=self._audio_channel_by_role["chords"])
                self.audio.flush()

        with self._state_lock:
            self.active_melody_nodes.clear()
            self.active_chord_nodes.clear()
            self.active_melody_note_counts.clear()
            self.active_chord_note_counts.clear()
            self._playback_done = False
            self._dirty_visual = True
            self._needs_full_redraw = True
        self._apply_highlight()

    def _read_bpm(self) -> float | None:
        try:
            bpm = float(self.bpm_box.text.strip())
            if bpm <= 0:
                return None
            return bpm
        except Exception:
            return None

    def _playback_loop(self, events: list[tuple[float, str, MidiEvent]], start_time: float):
        idx = 0
        processed = 0

        while idx < len(events) and not self._stop_event.is_set():
            now = time.perf_counter() - start_time
            dispatched = False

            while idx < len(events) and events[idx][0] <= now:
                _, role, evt = events[idx]
                self._dispatch_event(role, evt)
                idx += 1
                processed += 1
                dispatched = True

            if processed >= 16:
                if self.audio:
                    with self._audio_lock:
                        self.audio.flush()
                processed = 0

            if idx >= len(events):
                break

            if not dispatched:
                # Sleep in short bursts for accurate audio scheduling at high tempos.
                next_t = events[idx][0]
                remaining = next_t - now
                if remaining > 0.004:
                    time.sleep(min(remaining - 0.001, 0.004))
                elif remaining > 0.001:
                    time.sleep(0.0005)

        if processed > 0:
            if self.audio:
                with self._audio_lock:
                    self.audio.flush()
        with self._state_lock:
            self._playback_done = True

    def _dispatch_event(self, role: str, e: MidiEvent):
        node = e.note - MIN_NOTE

        target_nodes = self.active_melody_nodes if role == "melody" else self.active_chord_nodes
        target_counts = (
            self.active_melody_note_counts if role == "melody" else self.active_chord_note_counts
        )
        out_channel = self._audio_channel_by_role[role]
        changed = False
        if e.kind == "on":
            with self._state_lock:
                was_active = e.note in target_counts
                # Treat each pitch as active/inactive per role to avoid stale highlights
                # when source MIDI has mismatched repeated note_on/note_off pairs.
                target_counts[e.note] = 1
                if not was_active and node in self.node_to_i and MIN_NOTE <= e.note <= MAX_NOTE:
                    target_nodes.add(node)
                    changed = True
            if self.audio:
                vel = int(e.vel) if getattr(e, "vel", None) is not None else 80
                with self._audio_lock:
                    self.audio.note_on(e.note, vel, channel=out_channel)
        else:
            do_note_off = False
            with self._state_lock:
                if e.note in target_counts:
                    target_counts.pop(e.note, None)
                    do_note_off = True
                    if node in self.node_to_i and MIN_NOTE <= e.note <= MAX_NOTE:
                        target_nodes.discard(node)
                        changed = True
            if self.audio and do_note_off:
                with self._audio_lock:
                    self.audio.note_off(e.note, channel=out_channel)

        if changed:
            with self._state_lock:
                self._dirty_visual = True

    def _on_render_tick(self):
        if not self.is_playing:
            return

        should_redraw = False
        with self._state_lock:
            if self._dirty_visual:
                should_redraw = True
                self._dirty_visual = False

        if should_redraw:
            self._apply_highlight()

        with self._state_lock:
            done = self._playback_done
            active_empty = (
                not self.active_melody_note_counts and not self.active_chord_note_counts
            )
        if done and active_empty:
            self.stop()

    def _apply_highlight(self):
        with self._state_lock:
            melody_active = [n for n in self.active_melody_nodes if n in self.node_to_i]
            chord_active = [n for n in self.active_chord_nodes if n in self.node_to_i]
            needs_full_redraw = self._needs_full_redraw
            if needs_full_redraw:
                self._needs_full_redraw = False

        if not melody_active:
            self.melody_artist.set_offsets(np.empty((0, 2)))
            self.melody_artist.set_sizes([])
        else:
            m_idxs = [self.node_to_i[n] for n in melody_active]
            m_offsets = self.xy[m_idxs]
            # Smaller inner red ring when melody/chord share the same note.
            m_sizes = self.base_sizes[m_idxs] * 0.82
            self.melody_artist.set_offsets(m_offsets)
            self.melody_artist.set_sizes(m_sizes)

        if not chord_active:
            self.chord_artist.set_offsets(np.empty((0, 2)))
            self.chord_artist.set_sizes([])
        else:
            c_idxs = [self.node_to_i[n] for n in chord_active]
            c_offsets = self.xy[c_idxs]
            c_sizes = self.base_sizes[c_idxs] * 1.16
            self.chord_artist.set_offsets(c_offsets)
            self.chord_artist.set_sizes(c_sizes)

        now = time.perf_counter()
        min_draw_dt = 1.0 / max(self.target_fps, 1)
        if now - self._last_draw >= min_draw_dt:
            self._last_draw = now
            canvas = self.fig.canvas
            if self._blit_ready and not needs_full_redraw:
                canvas.restore_region(self._full_bg)
                self.ax.draw_artist(self.melody_artist)
                self.ax.draw_artist(self.chord_artist)
                canvas.blit(self.fig.bbox)
                canvas.flush_events()
            else:
                canvas.draw_idle()

    def _on_draw(self, event):
        if event.canvas is self.fig.canvas:
            self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
            if self.fig.canvas.supports_blit:
                self._full_bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
                self._blit_ready = self._full_bg is not None
            else:
                self._full_bg = None
                self._blit_ready = False

    def close(self):
        if self.render_timer.callbacks:
            self.render_timer.stop()
        self._stop_event.set()
        if self._playback_thread and self._playback_thread.is_alive():
            self._playback_thread.join(timeout=0.25)
        if self.audio:
            self.audio.all_notes_off()
            self.audio.close()
