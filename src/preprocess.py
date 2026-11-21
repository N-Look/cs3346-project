"""Fixed flow-feature policy and a train-only imputer for both models."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.impute import SimpleImputer

from src.load_data import DAYS, PROCESSED_DIR, feature_columns

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
CLEANED_DIR = PROCESSED_DIR.parent / "cleaned"
# Bump when fixed cleaning rules change so existing files are rebuilt.
CLEANING_VERSION = b"1"


def source_stamp(source: Path) -> dict[bytes, bytes]:
    """Identify the converted input and cleaning policy used for a saved day."""
    stat = source.stat()
    return {
        b"source_stat": f"{stat.st_dev}:{stat.st_ino}:{stat.st_size}:{stat.st_mtime_ns}".encode(),
        b"cleaning_version": CLEANING_VERSION,
    }


def prepare_cleaned_files(directory: Path = PROCESSED_DIR) -> list[Path]:
    """Save fixed feature cleaning once per day, preserving the source row order."""
    cleaned_dir = CLEANED_DIR if directory == PROCESSED_DIR else directory / "cleaned"
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for day in DAYS:
        source = directory / f"{day}.parquet"
        output = cleaned_dir / source.name
        stamp = source_stamp(source)
        source_info = pq.read_metadata(source)
        columns = [
            name for name in feature_columns(source_info.schema.to_arrow_schema().names)
            if name not in REDUNDANT_FEATURES
        ]
        if not cleaned_file_matches(output, source_info.num_rows, columns, stamp):
            write_cleaned_day(source, output, columns, stamp)
        outputs.append(output)
    return outputs


def cleaned_file_matches(
    path: Path, row_count: int, columns: list[str], stamp: dict[bytes, bytes]
) -> bool:
    if not path.exists():
        return False
    try:
        info = pq.read_metadata(path)
    except pa.ArrowInvalid:
        return False
    metadata = info.metadata or {}
    return (
        info.num_rows == row_count
        and info.schema.to_arrow_schema().names == columns
        and all(metadata.get(key) == value for key, value in stamp.items())
    )


def write_cleaned_day(
    source: Path, output: Path, columns: list[str], stamp: dict[bytes, bytes]
) -> None:
    frame = pd.read_parquet(source, columns=columns)
    matrix = frame.to_numpy(dtype=np.float32, copy=True)
    matrix[~np.isfinite(matrix)] = np.nan
    for name in NONNEGATIVE_FEATURES:
        values = matrix[:, columns.index(name)]
        values[values < 0] = np.nan
    # Treat +0 and -0 as the same value in duplicate checks.
    matrix[matrix == 0] = 0.0
    table = pa.Table.from_pandas(pd.DataFrame(matrix, columns=columns), preserve_index=False)
    table = table.replace_schema_metadata(stamp)
    with NamedTemporaryFile(dir=output.parent, prefix=f".{output.stem}-", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        pq.write_table(table, temporary, compression="zstd")
        if source_stamp(source) != stamp:
            raise RuntimeError(f"{source} changed during cleaning. Rerun preprocessing.")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def load_experiment_inputs(
    directory: Path = PROCESSED_DIR,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Load saved feature values and aligned split metadata in weekday/file row order."""
    paths = prepare_cleaned_files(directory)
    files = [pq.read_metadata(path) for path in paths]
    columns = files[0].schema.to_arrow_schema().names
    matrix = np.empty((sum(file.num_rows for file in files), len(columns)), dtype=np.float32)
    metadata = []
    start = 0
    for path, info in zip(paths, files, strict=True):
        source = directory / path.name
        features = pd.read_parquet(path)
        labels = pd.read_parquet(source, columns=list(SPLIT_COLUMNS))
        if (
            list(features.columns) != columns
            or len(features) != info.num_rows
            or len(labels) != info.num_rows
            or any((info.metadata or {}).get(key) != value
                   for key, value in source_stamp(source).items())
        ):
            raise ValueError(f"{path.name}: cleaned features and split metadata are out of sync.")
        end = start + info.num_rows
        matrix[start:end] = features.to_numpy(dtype=np.float32)
        metadata.append(labels)
        start = end
    return pd.concat(metadata, ignore_index=True), matrix, columns


def fit_imputer(matrix: np.ndarray, benign_train_rows: np.ndarray) -> SimpleImputer:
    """Fit shared median values on benign training rows, preserving all columns."""
    return SimpleImputer(strategy="median", keep_empty_features=True).fit(
        matrix[benign_train_rows]
    )


if __name__ == "__main__":
    for path in prepare_cleaned_files():
        print(path)
