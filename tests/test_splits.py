"""Regression checks for held-out flows and chronological partition boundaries."""

import numpy as np
import pandas as pd

from src.preprocess import fit_imputer
from src.splits import (
    EXCLUDED, TEST, TRAIN, VALIDATION, chronological_partitions, feature_fingerprints,
    holdout_partitions,
)


def test_same_second_stays_together_and_attempts_are_excluded():
    metadata = pd.DataFrame({
        "day": ["Monday"] * 7,
        "attack_type": ["BENIGN"] * 6 + ["Target"],
        "is_attempted": [False] * 6 + [True],
        "Timestamp": pd.to_datetime(
            ["2026-01-01 00:00:0" + str(i) for i in (0, 1, 2, 3, 3, 5, 6)]
        ),
    })
    assert chronological_partitions(metadata).tolist() == [
        TRAIN, TRAIN, TRAIN, TRAIN, TRAIN, TEST, EXCLUDED,
    ]


def test_all_heldout_rows_stay_in_test_and_test_copies_are_removed():
    metadata = pd.DataFrame({
        "attack_type": ["BENIGN"] * 6 + ["Target"] * 3,
        "is_attack": [False] * 6 + [True, True, False],
    })
    base = np.array([TRAIN, TRAIN, VALIDATION, VALIDATION, TEST, TRAIN,
                     TRAIN, VALIDATION, EXCLUDED])
    matrix = np.array([[1], [2], [3], [2], [4], [3], [2], [5], [6]], dtype=np.float32)
    fingerprints = feature_fingerprints(matrix)
    result = holdout_partitions(metadata, base, fingerprints, "Target")
    assert result.tolist() == [
        TRAIN, EXCLUDED, VALIDATION, EXCLUDED, TEST, EXCLUDED, TEST, TEST, EXCLUDED,
    ]
    assert not set(fingerprints[result == TRAIN]) & set(fingerprints[result == TEST])


def test_imputer_uses_only_benign_train_rows():
    matrix = np.array([[2], [4], [100], [np.nan]], dtype=np.float32)
    imputer = fit_imputer(matrix, np.array([0, 1]))
    assert imputer.transform(matrix[[3]])[0, 0] == 3
