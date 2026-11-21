"""Isolation Forest fitting and validation-only model selection."""

import numpy as np
from joblib import parallel_config
from sklearn.ensemble import IsolationForest

from src.evaluate import calibrate_threshold, detection_rate

N_ESTIMATORS = 100
CANDIDATES = ((256, 1.0), (1024, 1.0), (1024, 0.75))
BENIGN_TUNING_CAP = 20_000
ATTACK_TUNING_CAP = 2_000
TARGET_FPR = 0.01


def sample_rows(rows: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    """Use all rows below the cap, otherwise sample without replacement."""
    if len(rows) <= cap:
        return rows
    return np.sort(rng.choice(rows, cap, replace=False))


class FlowIsolationForest:
    """Fit on benign rows and return scores where larger means more anomalous."""

    def __init__(self, seed: int, max_samples: int, max_features: float, jobs: int):
        self.jobs = jobs
        self.model = IsolationForest(
            n_estimators=N_ESTIMATORS,
            max_samples=max_samples,
            max_features=max_features,
            random_state=seed,
            n_jobs=jobs,
        )

    def fit(self, benign_train: np.ndarray) -> "FlowIsolationForest":
        self.model.fit(benign_train)
        return self

    def score(self, matrix: np.ndarray) -> np.ndarray:
        with parallel_config(backend="threading", n_jobs=self.jobs):
            return -self.model.score_samples(matrix)


def sample_tuning_validation(
    benign_validation: np.ndarray,
    known_validation: dict[str, np.ndarray],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Use the same capped validation sample for both models' parameter search."""
    if not known_validation:
        raise ValueError("Tuning requires at least one known attack type in validation.")
    rng = np.random.default_rng(seed)
    benign_rows = sample_rows(np.arange(len(benign_validation)), BENIGN_TUNING_CAP, rng)
    benign_sample = benign_validation[benign_rows]
    attack_samples = [
        rows[sample_rows(np.arange(len(rows)), ATTACK_TUNING_CAP, rng)]
        for rows in known_validation.values()
    ]
    validation_sample = np.concatenate([benign_sample, *attack_samples])
    sample_sizes = [len(benign_sample), *(len(rows) for rows in attack_samples)]
    return validation_sample, np.cumsum(sample_sizes[:-1])


def tuning_recall(scores: np.ndarray, boundaries: np.ndarray) -> float:
    """Calculate known-attack macro recall at a calibrated 1% validation FPR."""
    benign_scores, *attack_scores = np.split(scores, boundaries)
    threshold = calibrate_threshold(benign_scores, TARGET_FPR)
    return float(np.mean([detection_rate(values, threshold) for values in attack_scores]))


def tune_model(
    benign_train: np.ndarray,
    benign_validation: np.ndarray,
    known_validation: dict[str, np.ndarray],
    seed: int,
    jobs: int,
) -> tuple[FlowIsolationForest, list[dict]]:
    """Choose settings by macro recall on known validation attacks at 1% FPR."""
    validation_sample, boundaries = sample_tuning_validation(
        benign_validation, known_validation, seed
    )
    scores = []
    best_model = None
    best_recall = -1.0
    for max_samples, max_features in CANDIDATES:
        samples_per_tree = min(max_samples, len(benign_train))
        model = FlowIsolationForest(seed, samples_per_tree, max_features, jobs).fit(benign_train)
        macro_recall = tuning_recall(model.score(validation_sample), boundaries)
        scores.append({
            "n_estimators": N_ESTIMATORS,
            "max_samples": model.model.max_samples_,
            "max_features": max_features,
            "validation_known_macro_recall": macro_recall,
        })
        if macro_recall > best_recall:
            best_model = model
            best_recall = macro_recall

    return best_model, scores
