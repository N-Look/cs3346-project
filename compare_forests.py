"""Compare the two forests on matching holdouts and seeds."""

import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd

from run_iforest import MIN_ATTACK_COUNT
from src.evaluate import MACRO_LABEL, compare_models, summarize
from src.iforest import TARGET_FPR
from src.load_data import REPO_ROOT

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def paired_comparison(isolation: pd.DataFrame, random: pd.DataFrame) -> dict:
    keys = ["heldout_type", "seed"]
    counts = ["n_attack_test", "n_benign_test"]
    isolation_counts = isolation.set_index(keys)[counts].sort_index()
    random_counts = random.set_index(keys)[counts].sort_index()
    if not isolation_counts.equals(random_counts):
        raise ValueError("The two result files must cover identical holdouts, seeds, and test counts.")
    summary = summarize(pd.concat([isolation, random], ignore_index=True), MIN_ATTACK_COUNT)
    comparison = compare_models(summary, target_fpr=TARGET_FPR)
    table = comparison["table"]
    table["n_attack_test_per_seed"] = isolation.groupby("heldout_type")["n_attack_test"].first()
    table.loc[MACRO_LABEL, "n_attack_test_per_seed"] = isolation_counts.groupby(level="seed")[
        "n_attack_test"
    ].sum().iloc[0]
    table["n_attack_test_per_seed"] = table["n_attack_test_per_seed"].astype(int)
    for name, runs in [("IsolationForest", isolation), ("RandomForest", random)]:
        per_type = runs.groupby("heldout_type")["fpr"]
        table[f"fpr_min_{name}"] = per_type.min()
        table[f"fpr_max_{name}"] = per_type.max()
        table.loc[MACRO_LABEL, f"fpr_min_{name}"] = runs["fpr"].min()
        table.loc[MACRO_LABEL, f"fpr_max_{name}"] = runs["fpr"].max()
    return comparison


def plot_recall(table: pd.DataFrame, output: Path) -> None:
    """Plot recall and seed ranges with a marker for small test sets."""
    plot = table.drop(index=MACRO_LABEL).sort_values("detection_mean_RandomForest")
    labels = [
        f"{name} *" if uncertain else name
        for name, uncertain in zip(plot.index, plot["uncertain"], strict=True)
    ]
    fig, ax = plt.subplots(figsize=(9.5, 6.3), layout="constrained")
    positions = range(len(plot))
    for model, name, offset, color in [
        ("IsolationForest", "Isolation Forest", -0.18, "#176972"),
        ("RandomForest", "Random Forest", 0.18, "#AE4B16"),
    ]:
        mean = plot[f"detection_mean_{model}"] * 100
        low = plot[f"detection_min_{model}"] * 100
        high = plot[f"detection_max_{model}"] * 100
        ax.errorbar(mean, [i + offset for i in positions], xerr=[(mean - low).clip(lower=0),
                    (high - mean).clip(lower=0)], fmt="o", capsize=3, color=color, label=name)
    ax.set_yticks(list(positions), labels)
    ax.set_xlim(-2, 104)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("Held-out attack recall (%)\n* Fewer than 50 test flows")
    ax.set_title("Detection at a 1% validation false-positive target", loc="left", pad=14)
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(loc="lower right", title="Mean and range across seeds")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args()
    directory = args.results_dir
    isolation_features = json.loads((directory / "iforest_features.json").read_text())
    random_features = json.loads((directory / "random_forest_features.json").read_text())
    if isolation_features != random_features:
        raise ValueError("The models were evaluated using different feature lists.")

    isolation = pd.read_csv(directory / "iforest_results.csv")
    random = pd.read_csv(directory / "random_forest_results.csv")
    comparison = paired_comparison(isolation, random)
    table = comparison["table"]
    table.to_csv(directory / "forest_comparison.csv")
    plot_recall(table, directory / "forest_recall.png")
    print(f"IF minus RF macro recall: {comparison['macro_diff_pp']:+.2f} percentage points")
    print(f"Worst test FPR: IF {comparison['worst_fpr_a']:.2%}, RF {comparison['worst_fpr_b']:.2%}")
    print(f"Proposal success criterion met: {comparison['success']}")


if __name__ == "__main__":
    main()
