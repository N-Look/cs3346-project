"""Run the held-out attack experiment on the corrected flow files."""

import argparse
import json
from pathlib import Path
import re
from time import perf_counter

import numpy as np
import pandas as pd

from src.evaluate import calibrate_threshold, evaluate_heldout, false_positive_rate, summarize
from src.iforest import TARGET_FPR, sample_rows, tune_model
from src.load_data import BENIGN, DAYS, PROCESSED_DIR, REPO_ROOT
from src.preprocess import fit_imputer, load_experiment_inputs
from src.random_forest import tune_random_forest
from src.splits import (
    EXCLUDED, PARTITIONS, TEST, TRAIN, VALIDATION, chronological_partitions, feature_fingerprints,
    holdout_partitions, save_manifest,
)

TRAIN_PER_DAY = 10_000
MIN_ATTACK_COUNT = 50


def run_fold(metadata, matrix, partitions, attack_type, seeds, jobs,
             model_name="IsolationForest"):
    """Reuse the fold's preprocessing for each seed and yield per-run metrics."""
    benign = metadata["attack_type"].eq(BENIGN).to_numpy()
    attack_types = metadata["attack_type"].to_numpy()
    day = metadata["day"].to_numpy()
    train_rows = np.flatnonzero((partitions == TRAIN) & benign)
    validation = partitions == VALIDATION
    test = partitions == TEST

    imputer = fit_imputer(matrix, train_rows)
    known_training = None
    if model_name == "RandomForest":
        known_training = imputer.transform(matrix[(partitions == TRAIN) & ~benign])
    benign_validation = imputer.transform(matrix[validation & benign])
    known_validation = {
        label: imputer.transform(matrix[validation & (attack_types == label)])
        for label in sorted(set(attack_types[validation & ~benign]))
    }
    benign_test = imputer.transform(matrix[test & benign])
    attack_test = imputer.transform(matrix[test & (attack_types == attack_type)])

    for seed in seeds:
        start = perf_counter()
        rng = np.random.default_rng(seed)
        model_rows = np.concatenate([
            sample_rows(train_rows[day[train_rows] == weekday], TRAIN_PER_DAY, rng)
            for weekday in DAYS
        ])
        training = imputer.transform(matrix[model_rows])
        if model_name == "RandomForest":
            model, candidates = tune_random_forest(
                training, known_training, benign_validation, known_validation, seed, jobs
            )
            fit_settings = {
                "n_known_model_train": len(known_training),
                "max_depth": model.model.max_depth,
                "max_samples": model.model.max_samples,
                "class_weight": model.model.class_weight,
            }
        else:
            model, candidates = tune_model(training, benign_validation, known_validation, seed, jobs)
            fit_settings = {"max_samples": model.model.max_samples_}

        validation_scores = model.score(benign_validation)
        threshold = calibrate_threshold(validation_scores, TARGET_FPR)
        test_scores = model.score(np.concatenate([benign_test, attack_test]))
        row = evaluate_heldout(
            model_name, attack_type, seed, threshold,
            test_scores[:len(benign_test)], test_scores[len(benign_test):],
        )
        yield row | {
            "n_benign_train": len(train_rows),
            "n_benign_model_train": len(model_rows),
            "n_benign_validation": len(benign_validation),
            "validation_fpr": false_positive_rate(validation_scores, threshold),
            "n_estimators": model.model.n_estimators,
            "max_features": model.model.max_features,
            "validation_candidates": json.dumps(candidates),
            "runtime_seconds": round(perf_counter() - start, 3),
        } | fit_settings


def main(model_name="IsolationForest"):
    parser = argparse.ArgumentParser(description=f"Run the {model_name} experiment.")
    parser.add_argument("--holdout", action="append", help="One label. Repeat to select several.")
    parser.add_argument("--seed", action="append", type=int, help="Defaults to 42, 43, and 44.")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--split-dir", type=Path, default=REPO_ROOT / "splits" / "iforest")
    args = parser.parse_args()
    seeds = list(dict.fromkeys(args.seed or (42, 43, 44)))
    if args.jobs < 1 or min(seeds) < 0 or max(seeds) >= 2**32:
        parser.error("Use at least one job and seeds in the range 0 to 2**32 - 1.")

    metadata, matrix, features = load_experiment_inputs(args.data_dir)
    base = chronological_partitions(metadata)
    labels = sorted(metadata.loc[(base != EXCLUDED) & metadata["is_attack"], "attack_type"].unique())
    holdouts = list(dict.fromkeys(args.holdout or labels))
    invalid = set(holdouts) - set(labels)
    if invalid:
        parser.error(f"Unknown or excluded holdout labels: {sorted(invalid)}. Choose from {labels}.")

    fingerprints = feature_fingerprints(matrix)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.split_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for attack_type in holdouts:
        partitions = holdout_partitions(metadata, base, fingerprints, attack_type)
        slug = re.sub(r"[^a-z0-9]+", "-", attack_type.lower()).strip("-")
        manifest_path = args.split_dir / f"{slug}.parquet"
        if model_name == "RandomForest":
            manifest = pd.read_parquet(manifest_path)
            stored = pd.Categorical(manifest["partition"], categories=PARTITIONS).codes
            if (not np.array_equal(manifest["row_id"], np.arange(len(metadata)))
                    or not np.array_equal(stored, partitions)):
                raise ValueError(f"{manifest_path} differs from the expected split for {attack_type}.")
            partitions = stored
        else:
            save_manifest(partitions, manifest_path)
        for row in run_fold(metadata, matrix, partitions, attack_type, seeds, args.jobs, model_name):
            results.append(row)
            print(f"{attack_type}, seed {row['seed']}: "
                  f"recall={row['detection_rate']:.2%}, test FPR={row['fpr']:.2%}", flush=True)

    runs = pd.DataFrame(results)
    prefix = "random_forest" if model_name == "RandomForest" else "iforest"
    runs.to_csv(args.output_dir / f"{prefix}_results.csv", index=False)
    summarize(runs, min_attack_count=MIN_ATTACK_COUNT).to_csv(
        args.output_dir / f"{prefix}_summary.csv", index=False
    )
    (args.output_dir / f"{prefix}_features.json").write_text(json.dumps(features, indent=2) + "\n")


if __name__ == "__main__":
    main()
