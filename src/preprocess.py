"""Fixed flow-feature policy and a train-only imputer for both models."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from src.load_data import DAYS, RAW_DIR, feature_columns, load_day

# These exporter names repeat average packet-length measurements.
REDUNDANT_FEATURES = {
    "Fwd Segment Size Avg",
    "Bwd Segment Size Avg",
    "Average Packet Size",
}
NONNEGATIVE_FEATURES = (
    "Flow Duration", "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Max", "Flow IAT Min",
)
SPLIT_COLUMNS = ("day", "Timestamp", "attack_type", "is_attack", "is_attempted")


def load_experiment_inputs(
    directory: Path = RAW_DIR,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Clean daily source records in memory, preserving weekday and row order.

    Explicit directories of converted daily Parquet files are also supported.
    """
    use_csv = (directory == RAW_DIR or any(directory.glob("*-WorkingHours.csv"))
               or not (directory / f"{DAYS[0]}.parquet").is_file())
    blocks = []
    metadata = []
    columns = []
    for day in DAYS:
        if use_csv:
            records = load_day(directory / f"{day}-WorkingHours.csv", day)
        else:
            records = pd.read_parquet(directory / f"{day}.parquet")

        selected = [name for name in feature_columns(records.columns)
                    if name not in REDUNDANT_FEATURES]
        if columns and selected != columns:
            raise ValueError(f"{day}: feature columns differ from preceding days.")
        columns = selected
        values = records[columns].to_numpy(dtype=np.float32, copy=True)
        values[~np.isfinite(values)] = np.nan
        for name in NONNEGATIVE_FEATURES:
            column = values[:, columns.index(name)]
            column[column < 0] = np.nan
        values[values == 0] = 0.0  # Normalize signed zeros for duplicate checks.
        blocks.append(values)
        metadata.append(records[list(SPLIT_COLUMNS)])
        del records

    return pd.concat(metadata, ignore_index=True), np.concatenate(blocks), columns


def fit_imputer(matrix: np.ndarray, benign_train_rows: np.ndarray) -> SimpleImputer:
    """Fit shared median values on benign training rows, preserving all columns."""
    return SimpleImputer(strategy="median", keep_empty_features=True).fit(
        matrix[benign_train_rows]
    )
