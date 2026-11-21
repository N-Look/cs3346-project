"""Load the corrected CICIDS2017 (WTMC 2021) day CSVs and convert them to Parquet.

Nothing is dropped, imputed or selected here: every original column and every
row is kept. Cleaning and feature selection are later team decisions (see
``docs/evaluation_protocol.md``, open decisions 1 and 8).

Raw format (checked on all five files on 2026-11-20)
----------------------------------------------------
* 84 columns: 7 identifier/metadata columns (``ID_COLUMNS``), 76 numeric flow
  features, and ``Label``. Header names have no surrounding whitespace in
  these files, but names are stripped anyway.
* ``Timestamp`` looks like ``04/07/2017 01:53:44 PM``: **day/month/year**,
  12-hour clock with AM/PM, one-second resolution (``TIMESTAMP_FORMAT``).
  Day-first is confirmed by the weekday: 03/07/2017 is Monday 3 July 2017 and
  every file holds a single date that matches its weekday. No time zone is
  stated, so timestamps are kept timezone-naive.
* Missing feature values are written as ``NaN`` and infinite ones as
  ``Infinity``; both are parsed to float NaN / inf and kept as they are.
* Labels: ``BENIGN``, an attack name such as ``DoS Hulk``, or an attack name
  followed by ``" - Attempted"``. Attempted flows are never their own class:
  ``attack_type`` removes the suffix, and ``is_attack`` is False for them.

Dtypes
------
Floats are parsed with pandas' default C parser. In pandas 3.1 this is exact:
on Wednesday, every value matched Python's ``float()``. (The old
``float_precision`` argument is deprecated.) A feature is then stored as
float32 only if the cast is **lossless** on the data (``FLOAT32_FEATURES``):
these are the
integer-valued counts, lengths, flags and window sizes, which all stay below
2**24 = 16,777,216, so float32 represents them exactly. The other features stay
float64, for two reasons. First, integer microsecond durations and IATs (up to
1.2e8), byte totals (up to 6.6e8) and bulk rates (up to 1.4e9) are larger than
2**24, so float32 would round them; Flow Duration alone would lose about
14,600 distinct values. Second, means, standard deviations and rates such as
Flow Bytes/s (up to 6e8) and Flow Packets/s lose digits at about 6e-8 relative
error. That error is small, but it merges distinct values and would change
exact-duplicate counts. Nothing would overflow float32 (the largest finite
value is about 1.4e9), so overflow was not the issue. ``load_day`` checks that
the float32 cast is lossless and raises an error if it is not.
"""

from __future__ import annotations

import gc
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "DAYS",
    "ID_COLUMNS",
    "LABEL_COLUMN",
    "DERIVED_COLUMNS",
    "BENIGN",
    "ATTEMPTED_SUFFIX",
    "TIMESTAMP_FORMAT",
    "FLOAT32_FEATURES",
    "RAW_DIR",
    "PROCESSED_DIR",
    "feature_columns",
    "derive_label_columns",
    "load_day",
    "convert_day",
    "convert_all",
    "load_all",
]

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw" / "wtmc2021"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")

ID_COLUMNS = (
    "Flow ID",
    "Src IP",
    "Src Port",
    "Dst IP",
    "Dst Port",
    "Protocol",
    "Timestamp",
)
LABEL_COLUMN = "Label"
DERIVED_COLUMNS = ("day", "attack_type", "is_attempted", "is_attack")

BENIGN = "BENIGN"
ATTEMPTED_SUFFIX = " - Attempted"
TIMESTAMP_FORMAT = "%d/%m/%Y %I:%M:%S %p"

_STRING_COLUMNS = ("Flow ID", "Src IP", "Dst IP", "Timestamp", LABEL_COLUMN)
_INT_COLUMNS = {"Src Port": "int32", "Dst Port": "int32", "Protocol": "int16"}

#: Features stored as float32. The cast is exact on all five WTMC 2021 files:
#: they are integer-valued with max |value| < 2**24. All others stay float64.
FLOAT32_FEATURES = (
    "Total Fwd Packet",
    "Total Bwd packets",
    "Total Length of Fwd Packet",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Packet Length Min",
    "Packet Length Max",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWR Flag Count",
    "ECE Flag Count",
    "Fwd Bytes/Bulk Avg",
    "Fwd Packet/Bulk Avg",
    "Bwd Packet/Bulk Avg",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "FWD Init Win Bytes",
    "Bwd Init Win Bytes",
    "Fwd Act Data Pkts",
    "Fwd Seg Size Min",
)

_DAY_FROM_FILENAME = re.compile(r"^(Monday|Tuesday|Wednesday|Thursday|Friday)\b")


def feature_columns(columns: Iterable[str]) -> list[str]:
    """Return the numeric flow-feature columns (not identifiers, label or derived)."""
    excluded = set(ID_COLUMNS) | {LABEL_COLUMN} | set(DERIVED_COLUMNS)
    return [c for c in columns if c not in excluded]


def derive_label_columns(labels: pd.Series) -> pd.DataFrame:
    """Return ``attack_type``, ``is_attempted`` and ``is_attack`` for raw labels.

    * ``attack_type``: the label with a trailing ``" - Attempted"`` removed
      (``BENIGN`` stays ``BENIGN``).
    * ``is_attempted``: True iff the label ends with ``" - Attempted"``.
    * ``is_attack``: True iff the label is not ``BENIGN`` and not Attempted.
    """
    labels = labels.astype("str").str.strip()
    is_attempted = labels.str.endswith(ATTEMPTED_SUFFIX)
    attack_type = labels.where(
        ~is_attempted, labels.str.slice(0, -len(ATTEMPTED_SUFFIX))
    )
    if (is_attempted & (attack_type == BENIGN)).any():
        raise ValueError("Found 'BENIGN - Attempted', which is not an expected label.")
    is_attack = (labels != BENIGN) & ~is_attempted
    return pd.DataFrame(
        {
            "attack_type": attack_type.astype("str"),
            "is_attempted": is_attempted.astype(bool),
            "is_attack": is_attack.astype(bool),
        },
        index=labels.index,
    )


def _infer_day(path: Path) -> str:
    m = _DAY_FROM_FILENAME.match(path.name)
    if not m:
        raise ValueError(
            f"Cannot infer the day from file name {path.name!r}; pass day=... explicitly."
        )
    return m.group(1)


def load_day(path: str | Path, day: str | None = None) -> pd.DataFrame:
    """Load one day CSV with explicit dtypes and add the derived columns.

    Keeps every row and every original column, in their original order.
    ``Timestamp`` is parsed to ``datetime64`` with ``TIMESTAMP_FORMAT``. Then
    ``day``, ``attack_type``, ``is_attempted`` and ``is_attack`` are appended.
    """
    path = Path(path)
    day = day if day is not None else _infer_day(path)

    raw_names = list(pd.read_csv(path, nrows=0).columns)
    names = [str(c).strip() for c in raw_names]
    if len(set(names)) != len(names):
        raise ValueError(f"{path.name}: duplicate column names after stripping whitespace.")
    missing = [c for c in (*ID_COLUMNS, LABEL_COLUMN) if c not in names]
    if missing:
        raise ValueError(f"{path.name}: missing expected columns {missing}.")
    features = feature_columns(names)

    dtype: dict[str, str] = {c: "str" for c in _STRING_COLUMNS}
    dtype.update(_INT_COLUMNS)
    dtype.update({c: "float64" for c in features})
    na_values = {c: ["NaN", "nan", ""] for c in features}

    df = pd.read_csv(
        path,
        header=0,
        names=names,
        dtype=dtype,
        keep_default_na=False,
        na_values=na_values,
        engine="c",
        low_memory=False,
    )

    for c in features:
        if c in FLOAT32_FEATURES:
            v64 = df[c].to_numpy()
            v32 = v64.astype(np.float32)
            if not np.array_equal(v32.astype(np.float64), v64, equal_nan=True):
                raise ValueError(
                    f"{path.name}: casting {c!r} to float32 would change values; "
                    "remove it from FLOAT32_FEATURES."
                )
            df[c] = v32

    raw_ts = df["Timestamp"]
    parsed_ts = pd.to_datetime(raw_ts, format=TIMESTAMP_FORMAT)
    # Lossless check: formatting the parsed value back must give the original text.
    if not (parsed_ts.dt.strftime(TIMESTAMP_FORMAT) == raw_ts).all():
        raise ValueError(f"{path.name}: Timestamp does not round-trip through {TIMESTAMP_FORMAT!r}.")
    df["Timestamp"] = parsed_ts
    del raw_ts, parsed_ts
    df[LABEL_COLUMN] = df[LABEL_COLUMN].str.strip()

    df["day"] = day
    df["day"] = df["day"].astype("str")
    derived = derive_label_columns(df[LABEL_COLUMN])
    for c in derived.columns:
        df[c] = derived[c]
    return df


def convert_day(
    day: str,
    raw_dir: str | Path = RAW_DIR,
    out_dir: str | Path = PROCESSED_DIR,
) -> dict:
    """Convert ``<raw_dir>/<day>-WorkingHours.csv`` to ``<out_dir>/<day>.parquet``.

    Returns a small summary dict (rows, columns, output bytes).
    """
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = raw_dir / f"{day}-WorkingHours.csv"
    dst = out_dir / f"{day}.parquet"
    df = load_day(src, day=day)
    n_rows, n_cols = df.shape
    df.to_parquet(dst, engine="pyarrow", compression="zstd", index=False)
    del df
    gc.collect()
    return {
        "day": day,
        "csv": str(src.relative_to(REPO_ROOT)) if src.is_relative_to(REPO_ROOT) else str(src),
        "parquet": str(dst.relative_to(REPO_ROOT)) if dst.is_relative_to(REPO_ROOT) else str(dst),
        "rows": int(n_rows),
        "columns": int(n_cols),
        "csv_bytes": src.stat().st_size,
        "parquet_bytes": dst.stat().st_size,
    }


def convert_all(
    days: Sequence[str] = DAYS,
    raw_dir: str | Path = RAW_DIR,
    out_dir: str | Path = PROCESSED_DIR,
) -> list[dict]:
    """Convert each day in turn, one file in memory at a time."""
    return [convert_day(d, raw_dir, out_dir) for d in days]


def load_all(
    days: Sequence[str] = DAYS,
    processed_dir: str | Path = PROCESSED_DIR,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read the processed Parquet files for ``days`` and concatenate them in order.

    ``columns`` limits which columns are read, which saves memory. The result
    has a fresh RangeIndex.
    """
    processed_dir = Path(processed_dir)
    frames = []
    for d in days:
        p = processed_dir / f"{d}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"{p} not found; run convert_all() first.")
        frames.append(
            pd.read_parquet(p, engine="pyarrow", columns=list(columns) if columns else None)
        )
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    for info in convert_all():
        print(
            f"{info['day']:<10} rows={info['rows']:>8,} cols={info['columns']} "
            f"csv={info['csv_bytes'] / 1e6:7.1f} MB parquet={info['parquet_bytes'] / 1e6:6.1f} MB"
        )
