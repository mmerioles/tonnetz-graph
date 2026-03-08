"""
generate_dataset.py

Generates 350 biased random walk sequences of length 1000 on a Tonnetz graph
and saves them to a CSV file.

Each row in the CSV = one sequence of 1000 note values.
Values are node indices [0-47] or -1 for a rest.

Usage (from project root):
    uv run python -m scripts.generate_dataset
    uv run python -m scripts.generate_dataset --output data/sequences.csv --seed 42
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from tonnetz.gen.walk import biased_random_walk

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
NUM_SEQUENCES = 350
SEQUENCE_LENGTH = 1000
DEFAULT_OUTPUT = "data/sequences.csv"
N_NODES = 48


def make_adjacency_matrix(n: int = N_NODES, seed: int | None = None) -> np.ndarray:
    """Generate a reproducible random Tonnetz adjacency matrix."""
    rng = np.random.default_rng(seed)
    mat = rng.exponential(0.3, size=(n, n))
    mat = mat / mat.max()
    mat[mat < 0.2] = 0
    return mat


def generate_dataset(
    num_sequences: int = NUM_SEQUENCES,
    sequence_length: int = SEQUENCE_LENGTH,
    output_path: str = DEFAULT_OUTPUT,
    seed: int | None = 42,
) -> Path:
    """
    Generate `num_sequences` biased random walk sequences of `sequence_length`
    and write them to a CSV file.

    Parameters
    ----------
    num_sequences : int
        Number of sequences to generate (default 350).
    sequence_length : int
        Length of each sequence (default 1000).
    output_path : str
        Path to save the CSV file.
    seed : int | None
        Base random seed for reproducibility. Each sequence gets its own
        derived seed (seed + i) so sequences are distinct but reproducible.

    Returns
    -------
    Path
        Path to the saved CSV file.
    """
    # Build one shared adjacency matrix for all sequences
    adj = make_adjacency_matrix(n=N_NODES, seed=seed)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_sequences} sequences of length {sequence_length}...")
    print(f"Graph: {N_NODES} nodes | Rest probability: 0.3 | Walk probability: 0.7")
    print(f"Output: {out.resolve()}\n")

    start = time.time()

    with open(out, "w", newline="") as f:
        writer = csv.writer(f)

        # Header: step_0, step_1, ..., step_999
        writer.writerow([f"step_{i}" for i in range(sequence_length)])

        for i in range(num_sequences):
            # Each sequence gets a unique but deterministic seed
            seq_seed = None if seed is None else seed + i
            sequence = biased_random_walk(
                adj_matrix=adj,
                length=sequence_length,
                seed=seq_seed,
            )
            writer.writerow(sequence)

            # Progress indicator every 50 sequences
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start
                print(f"  [{i + 1}/{num_sequences}] sequences generated ({elapsed:.1f}s)")

    elapsed = time.time() - start
    print(f"\nDone! {num_sequences} sequences saved to {out.resolve()}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"File size:  {out.stat().st_size / 1024:.1f} KB")

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Tonnetz walk dataset.")
    parser.add_argument("--num", type=int, default=NUM_SEQUENCES,
                        help=f"Number of sequences (default {NUM_SEQUENCES})")
    parser.add_argument("--length", type=int, default=SEQUENCE_LENGTH,
                        help=f"Length of each sequence (default {SEQUENCE_LENGTH})")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help=f"Output CSV path (default {DEFAULT_OUTPUT})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default 42)")
    args = parser.parse_args()

    generate_dataset(
        num_sequences=args.num,
        sequence_length=args.length,
        output_path=args.output,
        seed=args.seed,
    )