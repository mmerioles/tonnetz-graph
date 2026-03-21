"""
Evaluate LSTM generated sequences by comparing their distribution to training data
using KL divergence.
"""

import pandas as pd
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
import json
import ast


def parse_sequence(s):
    """Parse a sequence string that might be quoted or unquoted."""
    s = s.strip()
    # Remove surrounding quotes if present
    if s.startswith("'") and s.endswith("'"):
        s = s[1:-1]
    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1]
    # Parse the list
    return ast.literal_eval(s)


# Load training data
print("Loading training data...")
train_data = pd.read_csv("../data/lstm_data_multisong_16th.csv", low_memory=False)

# Convert string representations of lists to actual lists
train_sequences = []
for i, row in train_data.iterrows():
    try:
        # The 'x' column contains the sequences as strings
        seq = ast.literal_eval(row["x"])
        train_sequences.extend([int(x) for x in seq])
    except Exception as e:
        print(f"Error parsing training row {i}: {e}")
        continue

train_array = np.array(train_sequences, dtype=int)
print(f"Training data: {len(train_array)} total values")
print(f"Training data range: {train_array.min()} to {train_array.max()}")

# Load generated sequences
print("\nLoading generated sequences...")
generated_files = [
    "../data/lstm_generated_seq (5).csv",
    "../data/lstm_generated_seq (6).csv",
    "../data/lstm_generated_seq (8).csv",
]

all_generated_sequences = []
for file_path in generated_files:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    seq = parse_sequence(line)
                    all_generated_sequences.extend([int(x) for x in seq])
                except Exception as e:
                    print(f"Error at line {line_num} in {file_path}: {e}")
                    continue
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        continue

gen_array = np.array(all_generated_sequences, dtype=int)
if len(gen_array) > 0:
    print(
        f"Total generated data: {len(gen_array)} total values, range {gen_array.min()} to {gen_array.max()}"
    )
else:
    print(f"Total generated data: No data parsed!")

# Create histograms with bins 0-50
bins = np.arange(0, 51)  # 0 to 50 inclusive

# Compute histograms
train_hist, _ = np.histogram(train_array, bins=bins)
train_hist = train_hist / np.sum(train_hist)  # Normalize to probability distribution

print("\n" + "=" * 60)
print("KL DIVERGENCE ANALYSIS")
print("=" * 60)

if len(gen_array) == 0:
    print("No generated data to compare")
else:
    gen_hist, _ = np.histogram(gen_array, bins=bins)
    gen_hist = gen_hist / np.sum(gen_hist)  # Normalize to probability distribution

    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    train_hist_safe = np.clip(train_hist, epsilon, 1)
    gen_hist_safe = np.clip(gen_hist, epsilon, 1)

    # KL divergence: D_KL(P || Q) = sum(P * log(P / Q))
    kl_div = np.sum(train_hist_safe * np.log(train_hist_safe / gen_hist_safe))

    print(f"\nGenerated Data")
    print(f"  KL Divergence: {kl_div:.6f}")

    # Plot histograms
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Training data histogram
    axes[0].bar(
        bins[:-1],
        train_hist,
        width=0.8,
        edgecolor="black",
        alpha=0.7,
        color="steelblue",
    )
    axes[0].set_title("Training Data Distribution", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Value", fontsize=11)
    axes[0].set_ylabel("Probability", fontsize=11)
    axes[0].set_xlim(0, 50)
    axes[0].grid(True, alpha=0.3)

    # Generated data histogram
    axes[1].bar(
        bins[:-1], gen_hist, width=0.8, edgecolor="black", alpha=0.7, color="coral"
    )
    axes[1].set_title(
        f"Generated Data\nKL Divergence: {kl_div:.6f}", fontsize=13, fontweight="bold"
    )
    axes[1].set_xlabel("Value", fontsize=11)
    axes[1].set_ylabel("Probability", fontsize=11)
    axes[1].set_xlim(0, 50)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("lstm_evaluation.png", dpi=150, bbox_inches="tight")
    print("\nVisualization saved as 'lstm_evaluation.png'")
    plt.show()
