"""Supervised Random Forest baseline trained on benign and known attack flows."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.iforest import sample_tuning_validation, tuning_recall

N_ESTIMATORS = 100
DEPTHS = (12, 20, None)
MAX_SAMPLES_PER_TREE = 12_000
CLASS_WEIGHT = "balanced_subsample"


class FlowRandomForest:
    def __init__(self, seed: int, max_depth: int | None, max_samples: int, jobs: int):
        self.model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=max_depth,
            max_samples=max_samples,
            class_weight=CLASS_WEIGHT,
            random_state=seed,
            n_jobs=jobs,
        )

    def fit(self, training: np.ndarray, is_attack: np.ndarray) -> "FlowRandomForest":
        if not is_attack.any() or is_attack.all():
            raise ValueError("Random Forest requires both benign and known-attack training rows.")
        self.model.fit(training, is_attack)
        return self

    def score(self, matrix: np.ndarray) -> np.ndarray:
        """Return the predicted probability of the attack class."""
        return self.model.predict_proba(matrix)[:, 1]


def tune_random_forest(
    benign_train: np.ndarray,
    known_train: np.ndarray,
    benign_validation: np.ndarray,
    known_validation: dict[str, np.ndarray],
    seed: int,
    jobs: int,
) -> tuple[FlowRandomForest, list[dict]]:
    validation_sample, boundaries = sample_tuning_validation(
        benign_validation, known_validation, seed
    )
    training = np.concatenate([benign_train, known_train])
    is_attack = np.arange(len(training)) >= len(benign_train)
    max_samples = min(MAX_SAMPLES_PER_TREE, len(training))
    candidates = []
    best_model = None
    best_recall = -1.0
    for depth in DEPTHS:
        model = FlowRandomForest(seed, depth, max_samples, jobs).fit(training, is_attack)
        macro_recall = tuning_recall(model.score(validation_sample), boundaries)
        candidates.append({"max_depth": depth, "validation_known_macro_recall": macro_recall})
        if macro_recall > best_recall:
            best_model = model
            best_recall = macro_recall
    return best_model, candidates
