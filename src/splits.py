"""Reproducible chronological partitions for the leave-one-type-out experiment."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.load_data import BENIGN

# The corrected dataset's authors advise against using this label as malicious.
EXCLUDED_ATTACK_TYPES = {"DoS Hulk"}
EXCLUDED, TRAIN, VALIDATION, TEST = range(4)
PARTITIONS = ("excluded", "train", "validation", "test")


def feature_fingerprints(matrix: np.ndarray) -> np.ndarray:
    """Hash the cleaned float32 inputs before any data-dependent imputation."""
    return pd.util.hash_pandas_object(pd.DataFrame(matrix, copy=False), index=False).to_numpy()


def chronological_partitions(metadata: pd.DataFrame) -> np.ndarray:
    """Assign each day/label to contiguous 60/20/20 row blocks by start time."""
    eligible = ~metadata["is_attempted"] & ~metadata["attack_type"].isin(EXCLUDED_ATTACK_TYPES)
    rows = metadata.loc[eligible].sort_values("Timestamp", kind="stable")
    if rows["Timestamp"].isna().any():
        raise ValueError("Chronological splitting requires a timestamp for every eligible flow.")

    partitions = np.full(len(metadata), EXCLUDED, dtype=np.uint8)
    for _, group in rows.groupby(["day", "attack_type"], sort=False):
        # All flows with the same timestamp in this day/label stay together.
        rank = group["Timestamp"].rank(method="min").to_numpy() - 1
        positions = rank / len(group)
        partitions[group.index] = np.select(
            [positions < 0.6, positions < 0.8], [TRAIN, VALIDATION], default=TEST
        )
    return partitions


def holdout_partitions(
    metadata: pd.DataFrame,
    base: np.ndarray,
    fingerprints: np.ndarray,
    attack_type: str,
) -> np.ndarray:
    """Reserve every eligible held-out flow, then purge cross-partition copies."""
    heldout = metadata["is_attack"].to_numpy() & metadata["attack_type"].eq(attack_type).to_numpy()
    if not heldout.any() or attack_type in EXCLUDED_ATTACK_TYPES:
        raise ValueError(f"Unknown or excluded holdout: {attack_type!r}")

    partitions = base.copy()
    partitions[heldout] = TEST
    # Preserve the test distribution and remove matching rows from earlier
    # partitions. Matching validation rows similarly take priority over train.
    for earlier in (VALIDATION, TRAIN):
        blocked = np.unique(fingerprints[partitions > earlier])
        rows = np.flatnonzero(partitions == earlier)
        partitions[rows[np.isin(fingerprints[rows], blocked)]] = EXCLUDED

    for partition, name in enumerate(PARTITIONS[1:], start=TRAIN):
        if not np.any((partitions == partition) & metadata["attack_type"].eq(BENIGN).to_numpy()):
            raise ValueError(f"{attack_type}: no benign {name} rows remain.")
    return partitions


def save_manifest(partitions: np.ndarray, path: Path) -> None:
    """Save row IDs and partition names, including excluded rows."""
    manifest = pd.DataFrame({
        "row_id": np.arange(len(partitions)),
        "partition": pd.Categorical.from_codes(partitions, PARTITIONS),
    })
    temporary = path.with_suffix(".tmp.parquet")
    manifest.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)
