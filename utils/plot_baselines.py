"""Plot shaded convergence curves for MAPPO and IPPO baselines.

Reads CSV metric files produced by multi-seed training runs and generates
shaded plots (mean ± std) for key metrics.

Usage:
    python -m llm_toc.utils.plot_baselines \
        --csv runs/metrics.csv \
        --output-dir runs/figures
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------
def load_metrics(csv_path: Path) -> Dict[str, Dict[int, pd.DataFrame]]:
    """Load metrics CSV and group by algorithm and seed.

    The train.py script writes a single metrics.csv with columns:
        run_name, step, key, value

    Each run_name encodes algorithm and seed (e.g. ``llm_toc_mappo_seed42``).
    We return a nested dict: {algo_name: {seed: DataFrame}}.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"No metrics.csv at {csv_path}")

    df = pd.read_csv(csv_path)
    result: Dict[str, Dict[int, pd.DataFrame]] = {}

    for run_name, group in df.groupby("run_name"):
        # Parse run_name: llm_toc_{algo}_seed{seed}
        # e.g., "llm_toc_mappo_seed42" -> algo="mappo", seed=42
        parts = str(run_name).split("_")
        if len(parts) < 4 or parts[0] != "llm" or parts[1] != "toc":
            continue
        algo_name = parts[2]  # "mappo" or "ippo"
        seed_part = parts[3]  # "seed42"
        if not seed_part.startswith("seed"):
            continue
        try:
            seed = int(seed_part[4:])  # extract number after "seed"
        except ValueError:
            continue

        # Pivot to wide format: step as index, keys as columns
        wide = group.pivot_table(
            index="step", columns="key", values="value", aggfunc="mean"
        ).sort_index()

        if algo_name not in result:
            result[algo_name] = {}
        result[algo_name][seed] = wide

    return result


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate_metric(
    seeds: Dict[int, pd.DataFrame],
    metric_key: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate a single metric across seeds.

    Aligns all seeds to a common step grid (union of all steps), interpolates
    missing values, then computes mean and std.

    Returns:
        Tuple of (steps, mean, std) arrays.
    """
    if not seeds:
        return np.array([]), np.array([]), np.array([])

    # Collect per-seed series, reindexed to common steps.
    all_steps: Optional[np.ndarray] = None
    series_list: List[pd.Series] = []

    for seed, df in seeds.items():
        if metric_key not in df.columns:
            continue
        s = df[metric_key].dropna()
        if len(s) == 0:
            continue
        series_list.append(s)
        if all_steps is None:
            all_steps = s.index.to_numpy()
        else:
            all_steps = np.union1d(all_steps, s.index.to_numpy())

    if not series_list or all_steps is None:
        return np.array([]), np.array([]), np.array([])

    # Reindex each series to common steps with forward-fill then back-fill.
    aligned = []
    for s in series_list:
        reindexed = s.reindex(all_steps).ffill().bfill()
        aligned.append(reindexed.to_numpy())

    matrix = np.stack(aligned, axis=0)  # (num_seeds, num_steps)
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)
    return all_steps, mean, std


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_shaded(
    ax: plt.Axes,
    steps: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    label: str,
    color: str,
    alpha_fill: float = 0.2,
    linewidth: float = 2.0,
) -> None:
    """Plot a mean line with shaded std band."""
    ax.plot(steps, mean, color=color, label=label, linewidth=linewidth)
    ax.fill_between(
        steps,
        mean - std,
        mean + std,
        color=color,
        alpha=alpha_fill,
    )


def generate_plots(
    csv_path: Path,
    output_dir: Path,
) -> List[Path]:
    """Generate shaded convergence comparison plots.

    Produces:
        1. collective_return.png — Episode return (mean ± std)
        2. success_rate.png — Success rate (mean ± std)
        3. episode_length.png — Episode length (mean ± std)
        4. value_loss.png — Value loss (mean ± std)
        5. entropy.png — Policy entropy (mean ± std)

    Args:
        csv_path: Path to the combined metrics.csv file.
        output_dir: Where to save the PNG files.

    Returns:
        List of generated figure paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_data = load_metrics(csv_path)
    mappo_seeds = all_data.get("mappo", {})
    ippo_seeds = all_data.get("ippo", {})

    print(f"MAPPO seeds loaded: {sorted(mappo_seeds.keys())}")
    print(f"IPPO seeds loaded:  {sorted(ippo_seeds.keys())}")

    # Metrics to plot: (csv_key, title, ylabel, filename)
    metrics = [
        ("train/collective_return", "Collective Return", "Return", "collective_return.png"),
        ("train/success_rate", "Success Rate", "Rate", "success_rate.png"),
        ("train/episode_length", "Episode Length", "Steps", "episode_length.png"),
        ("train/agent0/value_loss", "Value Loss (Agent 0)", "Loss", "value_loss.png"),
        ("train/agent0/entropy", "Policy Entropy (Agent 0)", "Entropy", "entropy.png"),
    ]

    colors = {"MAPPO": "#2196F3", "IPPO": "#FF5722"}
    fig_paths: List[Path] = []

    for key, title, ylabel, filename in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))

        for algo_name, seeds, color in [
            ("MAPPO", mappo_seeds, colors["MAPPO"]),
            ("IPPO", ippo_seeds, colors["IPPO"]),
        ]:
            steps, mean, std = aggregate_metric(seeds, key)
            if len(steps) == 0:
                print(f"  [skip] {algo_name} has no data for {key}")
                continue
            plot_shaded(ax, steps, mean, std, label=algo_name, color=color)
            print(f"  {algo_name} {key}: {len(steps)} points, "
                  f"final mean={mean[-1]:.4f} ± {std[-1]:.4f}")

        ax.set_title(f"{title} — Baseline Convergence (3 seeds)", fontsize=14)
        ax.set_xlabel("Environment Steps", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(fontsize=12, loc="best")
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

        fig.tight_layout()
        path = output_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        fig_paths.append(path)
        print(f"  Saved: {path}")

    return fig_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot shaded convergence curves for MAPPO/IPPO baselines."
    )
    parser.add_argument(
        "--csv", type=Path, default=Path("runs/metrics.csv"),
        help="Path to the combined metrics.csv file",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/figures"),
        help="Where to save the PNG files",
    )
    args = parser.parse_args()

    paths = generate_plots(args.csv, args.output_dir)
    print(f"\n=== Generated {len(paths)} figures in {args.output_dir} ===")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
