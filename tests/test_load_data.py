"""Tests for src/load_data.py. Run with: .venv/bin/python -m pytest"""

import math

import numpy as np
import pandas as pd
import pytest

from src.load_data import (
    DAYS,
    PROCESSED_DIR,
    RAW_DIR,
    convert_day,
    derive_label_columns,
    load_all,
    load_day,
)

# A tiny CSV in the raw WTMC 2021 layout. It has the 7 identifier columns,
# a few feature columns and Label. Some header names have padding whitespace
# on purpose, and the NaN/Infinity tokens are the ones the real files use.
_HEADER = (
    "Flow ID, Src IP,Src Port, Dst IP,Dst Port,Protocol,Timestamp,"
    "Flow Duration,Total Fwd Packet,Flow Bytes/s, Label "
)
_ROWS = [
    # label,                                      flow bytes/s
    ("BENIGN", "12.5"),
    ("DoS Hulk", "NaN"),
    ("DoS Hulk - Attempted", "Infinity"),
    ("Web Attack - Brute Force", "1.0E7"),
    ("Web Attack - Brute Force - Attempted", "0.0"),
    ("Infiltration - Attempted", "3.25"),
    ("PortScan", "7"),
]
_EXPECTED = {
    # label: (attack_type, is_attempted, is_attack)
    "BENIGN": ("BENIGN", False, False),
    "DoS Hulk": ("DoS Hulk", False, True),
    "DoS Hulk - Attempted": ("DoS Hulk", True, False),
    "Web Attack - Brute Force": ("Web Attack - Brute Force", False, True),
    "Web Attack - Brute Force - Attempted": ("Web Attack - Brute Force", True, False),
    "Infiltration - Attempted": ("Infiltration", True, False),
    "PortScan": ("PortScan", False, True),
}


def _write_csv(path, rows=_ROWS, total_fwd="3"):
    lines = [_HEADER]
    for i, (label, fbs) in enumerate(rows):
        lines.append(
            f"10.0.0.1-10.0.0.2-{1000 + i}-80-6,10.0.0.1,{1000 + i},10.0.0.2,80,6,"
            f"04/07/2017 0{i + 1}:05:0{i} PM,{120000000 + i},{total_fwd},{fbs},{label}"
        )
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def tiny_csv(tmp_path):
    return _write_csv(tmp_path / "Tuesday-WorkingHours.csv")


# ------------------------------------------------------------ label derivation

def test_derive_label_columns_table():
    labels = pd.Series(list(_EXPECTED))
    out = derive_label_columns(labels)
    for i, lab in enumerate(labels):
        assert tuple(out.iloc[i]) == _EXPECTED[lab], lab


def test_attempted_is_never_its_own_class():
    out = derive_label_columns(pd.Series(list(_EXPECTED)))
    assert not out["attack_type"].str.contains("Attempted").any()
    # Every Attempted row maps to an attack type that also exists without the suffix.
    assert set(out.loc[out["is_attempted"], "attack_type"]) <= {
        "DoS Hulk", "Web Attack - Brute Force", "Infiltration"
    }
    assert not (out["is_attempted"] & out["is_attack"]).any()


def test_benign_attempted_rejected():
    with pytest.raises(ValueError):
        derive_label_columns(pd.Series(["BENIGN - Attempted"]))


# ------------------------------------------------------------ load_day on synthetic CSV

def test_load_day_derives_columns(tiny_csv):
    df = load_day(tiny_csv)
    assert len(df) == len(_ROWS)  # no rows dropped
    assert list(df.columns[-4:]) == ["day", "attack_type", "is_attempted", "is_attack"]
    assert (df["day"] == "Tuesday").all()
    for _, row in df.iterrows():
        exp = _EXPECTED[row["Label"]]
        assert (row["attack_type"], row["is_attempted"], row["is_attack"]) == exp
    assert df["is_attempted"].dtype == bool and df["is_attack"].dtype == bool


def test_load_day_strips_column_names_and_keeps_all_columns(tiny_csv):
    df = load_day(tiny_csv)
    assert "Label" in df.columns and "Src IP" in df.columns and "Dst IP" in df.columns
    assert all(c == c.strip() for c in df.columns)
    assert len(df.columns) == 11 + 4  # 11 original + 4 derived


def test_load_day_dtypes_and_values(tiny_csv):
    df = load_day(tiny_csv)
    assert df["Src Port"].dtype == np.int32 and df["Dst Port"].dtype == np.int32
    assert df["Protocol"].dtype == np.int16
    assert df["Total Fwd Packet"].dtype == np.float32  # lossless small integers
    assert df["Flow Duration"].dtype == np.float64  # microseconds above 2**24
    assert df["Flow Bytes/s"].dtype == np.float64
    # Large integer duration is kept exactly.
    assert df["Flow Duration"].iloc[0] == 120000000.0
    # NaN and Infinity tokens are kept, not imputed.
    fbs = df["Flow Bytes/s"]
    assert math.isnan(fbs.iloc[1]) and np.isposinf(fbs.iloc[2]) and fbs.iloc[3] == 1e7


def test_load_day_timestamp_is_day_first(tiny_csv):
    ts = load_day(tiny_csv)["Timestamp"]
    assert pd.api.types.is_datetime64_any_dtype(ts)
    # "04/07/2017 01:05:00 PM" is 4 July 2017 (a Tuesday), 13:05:00.
    assert ts.iloc[0] == pd.Timestamp("2017-07-04 13:05:00")
    assert (ts.dt.day_name() == "Tuesday").all()


def test_load_day_rejects_lossy_float32(tmp_path):
    path = _write_csv(tmp_path / "Monday-WorkingHours.csv", total_fwd="16777217")
    with pytest.raises(ValueError, match="float32"):
        load_day(path)


def test_load_day_needs_day_for_unknown_filename(tmp_path):
    path = _write_csv(tmp_path / "sample.csv")
    with pytest.raises(ValueError):
        load_day(path)
    assert (load_day(path, day="Friday")["day"] == "Friday").all()


def test_convert_and_load_all_roundtrip(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_csv(raw / "Tuesday-WorkingHours.csv")
    info = convert_day("Tuesday", raw_dir=raw, out_dir=tmp_path / "proc")
    assert info["rows"] == len(_ROWS)
    back = load_all(days=["Tuesday"], processed_dir=tmp_path / "proc")
    pd.testing.assert_frame_equal(back, load_day(raw / "Tuesday-WorkingHours.csv"))


# ------------------------------------------------------------ real processed files

def _csv_rows(path):
    """Count data rows (lines minus header) without parsing the CSV."""
    n, last = 0, b"\n"
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            n += chunk.count(b"\n")
            last = chunk[-1:]
    if last != b"\n":
        n += 1
    return n - 1


_have_real = all((PROCESSED_DIR / f"{d}.parquet").exists() and (RAW_DIR / f"{d}-WorkingHours.csv").exists()
                 for d in DAYS)


@pytest.mark.skipif(not _have_real, reason="raw CSVs or processed Parquet files not present")
@pytest.mark.parametrize("day", DAYS)
def test_parquet_rows_equal_csv_rows(day):
    import pyarrow.parquet as pq

    pq_rows = pq.ParquetFile(PROCESSED_DIR / f"{day}.parquet").metadata.num_rows
    assert pq_rows == _csv_rows(RAW_DIR / f"{day}-WorkingHours.csv")


@pytest.mark.skipif(not _have_real, reason="raw CSVs or processed Parquet files not present")
def test_real_derived_columns_consistent():
    df = load_all(columns=["Label", "day", "attack_type", "is_attempted", "is_attack"])
    assert len(df) == 2_100_814
    exp = derive_label_columns(df["Label"])
    pd.testing.assert_frame_equal(df[["attack_type", "is_attempted", "is_attack"]], exp)
    assert not df["attack_type"].str.contains("Attempted").any()
    assert df["day"].isin(DAYS).all()
