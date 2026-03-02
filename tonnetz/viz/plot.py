import os
import time
from pathlib import Path

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button, TextBox

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
except ImportError as _e:  # mido/fluidsynth/tonnetz.midi.player missing
    _AUDIO_AVAILABLE = False
    _AUDIO_IMPORT_ERROR = _e
from tonnetz.util.util import create_note_labels

MIN_NOTE = 36
MAX_NOTE = 83  # inclusive

_MODULE_DIR = Path(__file__).resolve().parent         
_PROJECT_ROOT = _MODULE_DIR.parents[2]                 
_MIDI_DIR = _PROJECT_ROOT / "tonnetz-graph" / "raw_midi"
_DEFAULT_SOUNDFONT = _MIDI_DIR / "GeneralUser-GS.sf2"

def plot_graph(
    input_graph: nx.DiGraph,
    show_isolated_nodes: bool = False,
    show: bool = True,
    name: str = "Network Graph",
    centralities: dict | None = None,
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
        midi_path = name
        if not os.path.isabs(midi_path):
            candidate = _MIDI_DIR / name
            if candidate.exists():
                midi_path = str(candidate)

        soundfont_path = str(_DEFAULT_SOUNDFONT) if _DEFAULT_SOUNDFONT.exists() else None

        if os.path.exists(midi_path):
            overlay = TonnetzRealtimeOverlay(
                fig=fig,
                ax=ax,
                node_artist=node_artist,
                nodes_in_graph=list(G.nodes()),
                node_pos=node_pos,
                midi_path=midi_path,
                soundfont_path=soundfont_path,
                target_channel=0,
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
        midi_path: str,
        soundfont_path: str | None = None,
        target_channel: int | None = None,
    ):
        self.fig = fig
        self.ax = ax
        self.node_artist = node_artist

        self.nodes = list(nodes_in_graph)
        self.node_to_i = {n: i for i, n in enumerate(self.nodes)}

        self.node_pos = node_pos
        self.xy = np.array([self.node_pos[n] for n in self.nodes], dtype=float)

        self.fig.canvas.draw()

        sizes = np.array(self.node_artist.get_sizes(), dtype=float)
        if sizes.size == 1 and len(self.nodes) > 1:
            sizes = np.repeat(sizes, repeats=len(self.nodes))
        self.base_sizes = sizes.copy()

        self.highlight_artist = ax.scatter(
            [],
            [],
            s=[],
            facecolors="none",
            edgecolors="lime",
            linewidths=2.5,
            zorder=10,
        )

        self.active_nodes: set[int] = set()

        self.events0 = midi_to_events_ticks(midi_path, target_channel=target_channel)
        self.base_bpm = float(get_initial_bpm(midi_path))
        self.events: list[MidiEvent] = self.events0[:]

        self.event_idx = 0

        self.audio = None
        if soundfont_path:
            self.audio = FluidSynthPlayer(soundfont_path)

        self.is_playing = False
        self.t0 = 0.0

        ax_play = fig.add_axes([0.85, 0.92, 0.12, 0.06])
        self.btn = Button(ax_play, "Play")

        ax_bpm = fig.add_axes([0.70, 0.92, 0.13, 0.06])
        self.bpm_box = TextBox(ax_bpm, "BPM", initial=f"{self.base_bpm:.2f}")

        self.btn.on_clicked(self._toggle_play)

        self.timer = fig.canvas.new_timer(interval=5)
        self.timer.add_callback(self._on_tick)

    def _toggle_play(self, _event):
        if not self.is_playing:
            self.start()
        else:
            self.stop()

    def start(self):
        self.events = self.events0[:]

        bpm = self._read_bpm()
        if bpm is not None and bpm > 0 and abs(bpm - self.base_bpm) > 1e-6:
            self.events = scale_events_bpm(self.events0, original_bpm=self.base_bpm, new_bpm=bpm)

        self.is_playing = True
        self.t0 = time.perf_counter()
        self.event_idx = 0

        self.active_nodes.clear()
        self._apply_highlight()

        self.btn.label.set_text("Stop")
        self.timer.start()

    def stop(self):
        self.is_playing = False
        self.timer.stop()
        self.btn.label.set_text("Play")

        if self.audio:
            for node in list(self.active_nodes):
                midi_note = node + MIN_NOTE
                self.audio.note_off(midi_note)
            self.audio.flush()

        self.active_nodes.clear()
        self._apply_highlight()

    def _read_bpm(self) -> float | None:
        try:
            bpm = float(self.bpm_box.text.strip())
            if bpm <= 0:
                return None
            return bpm
        except Exception:
            return None

    def _on_tick(self):
        if not self.is_playing:
            return

        t_now = time.perf_counter() - self.t0
        changed = False

        while self.event_idx < len(self.events) and self.events[self.event_idx].t <= t_now:
            e = self.events[self.event_idx]
            self.event_idx += 1

            if not (MIN_NOTE <= e.note <= MAX_NOTE):
                continue

            node = e.note - MIN_NOTE
            if node not in self.node_to_i:
                continue

            if e.kind == "on":
                if node not in self.active_nodes:
                    self.active_nodes.add(node)
                    changed = True
                if self.audio:
                    # Guard velocity
                    vel = int(e.vel) if getattr(e, "vel", None) is not None else 80
                    self.audio.note_on(e.note, vel)
            else:  # "off"
                if node in self.active_nodes:
                    self.active_nodes.remove(node)
                    changed = True
                if self.audio:
                    self.audio.note_off(e.note)

        if self.event_idx % 10 == 0 and self.audio:
            self.audio.flush()

        if changed:
            self._apply_highlight()

        if self.event_idx >= len(self.events) and not self.active_nodes:
            self.stop()

    def _apply_highlight(self):
        active = [n for n in self.active_nodes if n in self.node_to_i]
        if not active:
            self.highlight_artist.set_offsets(np.empty((0, 2)))
            self.highlight_artist.set_sizes([])
            self.fig.canvas.draw_idle()
            return

        idxs = [self.node_to_i[n] for n in active]

        offsets = self.xy[idxs]
        # Slightly larger ring (does NOT change base node sizes)
        sizes = self.base_sizes[idxs] * 1.10

        self.highlight_artist.set_offsets(offsets)
        self.highlight_artist.set_sizes(sizes)
        self.fig.canvas.draw_idle()

    def close(self):
        if self.timer.callbacks:
            self.timer.stop()
        if self.audio:
            self.audio.all_notes_off()
            self.audio.close()