"""Evaluate random-walk generation against an original MIDI distribution."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from tonnetz.gen.create_midi import create_midi_from_list
from tonnetz.gen.walk import biased_random_walk
from tonnetz.midi.parser import extract_timed_events, gen_transition_poly


# Editable evaluation settings
MIDI_FILENAME = "Knockin_on_Heaven_Door.mid"
MIDI_CHANNEL = 2
CENTRALITY = "eigenvector"  # "degree", "betweenness", or "eigenvector"
REST_PROB = 0.25
SEED = 47
BIN_MIN = 0
BIN_MAX = 47
SAVE_GENERATED_MIDI = False


def _normalize_hist(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges)
    total = counts.sum()
    if total == 0:
        return np.zeros_like(counts, dtype=float)
    return counts.astype(float) / float(total)


def _kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    p_safe = np.clip(p, epsilon, 1.0)
    q_safe = np.clip(q, epsilon, 1.0)
    return float(np.sum(p_safe * np.log(p_safe / q_safe)))


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    midi_path = os.path.join(project_root, "raw_midi", MIDI_FILENAME)

    print("Loading MIDI and building adjacency matrix...")
    transition_matrix = gen_transition_poly(midi_path, target_channel=MIDI_CHANNEL)
    if transition_matrix is None:
        raise RuntimeError("Failed to create transition matrix from MIDI.")

    print("Extracting original note sequence...")
    events = extract_timed_events(midi_path, target_channel=MIDI_CHANNEL)
    original_sequence = np.array(
        [evt["note"] for evt in events if evt["type"] == "on"], dtype=int
    )
    if original_sequence.size == 0:
        raise RuntimeError(
            "No note_on events found in original MIDI for the selected channel."
        )

    gen_length = int(original_sequence.size)
    print(f"Generating random-walk sequence (length={gen_length})...")
    generated_sequence = biased_random_walk(
        transition_matrix,
        length=gen_length,
        rest_prob=REST_PROB,
        centrality_type=CENTRALITY,
        seed=SEED,
    )
    generated_sequence = np.array(generated_sequence, dtype=int)

    # Align comparison domain to user-requested histogram bins.
    bin_min = int(BIN_MIN)
    bin_max = int(BIN_MAX)
    if bin_max < bin_min:
        raise ValueError("bin-max must be >= bin-min")

    # Integer-centered bin edges so both endpoints are included.
    bin_edges = np.arange(bin_min - 0.5, bin_max + 1.5, 1.0)

    original_in_range = original_sequence[
        (original_sequence >= bin_min) & (original_sequence <= bin_max)
    ]
    generated_in_range = generated_sequence[
        (generated_sequence >= bin_min) & (generated_sequence <= bin_max)
    ]

    print(f"Original notes in range [{bin_min}, {bin_max}]: {len(original_in_range)}")
    print(f"Generated notes in range [{bin_min}, {bin_max}]: {len(generated_in_range)}")

    orig_hist = _normalize_hist(original_in_range, bin_edges)
    gen_hist = _normalize_hist(generated_in_range, bin_edges)

    kl = _kl_divergence(orig_hist, gen_hist)

    print("\n" + "=" * 56)
    print("RANDOM WALK EVALUATION")
    print("=" * 56)
    print(f"MIDI: {MIDI_FILENAME}")
    print(f"Channel: {MIDI_CHANNEL}")
    print(f"Centrality: {CENTRALITY}")
    print(f"KL Divergence (Original || Generated): {kl:.6f}")

    if SAVE_GENERATED_MIDI:
        out_midi = os.path.join(project_root, "raw_midi", "rw_eval_generated.mid")
        create_midi_from_list(
            generated_sequence.tolist(),
            output_path=out_midi,
            bpm=88.0,
            channel=0,
            velocity=92,
            note_length_beats=0.5,
        )
        print(f"Generated MIDI saved to: {out_midi}")

    x_vals = np.arange(bin_min, bin_max + 1)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(
        x_vals,
        orig_hist,
        width=0.8,
        edgecolor="black",
        alpha=0.6,
        color="steelblue",
        label="Original MIDI",
    )
    ax.bar(
        x_vals,
        gen_hist,
        width=0.5,
        edgecolor="black",
        alpha=0.6,
        color="coral",
        label="Random Walk Generated",
    )
    ax.set_title(
        f"Original vs Random Walk Distribution (KL: {kl:.6f})", fontweight="bold"
    )
    ax.set_xlabel("Sequence Value")
    ax.set_ylabel("Probability")
    ax.set_xlim(bin_min - 0.5, bin_max + 0.5)
    ax.grid(True, alpha=0.25)
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
