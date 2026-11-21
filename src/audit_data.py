"""Data audit of the processed corrected-CICIDS2017 Parquet files.

This script only reports: it does not drop, impute or select anything.
Writes ``reports/data_audit.md`` (tables) and ``reports/data_audit.json``
(the same numbers in machine-readable form).

Run from the repo root:  .venv/bin/python -I src/audit_data.py

Memory: each day's Parquet file is processed in turn. Only small per-row
arrays (a 64-bit feature hash, the labels and the timestamps) and a ~200k-row
sample are kept across days.
"""

from __future__ import annotations

import gc
import json
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# With ``python -I`` the script directory is not on sys.path, so add the repo
# root explicitly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.load_data import (  # noqa: E402
    BENIGN,
    DAYS,
    FLOAT32_FEATURES,
    ID_COLUMNS,
    LABEL_COLUMN,
    PROCESSED_DIR,
    RAW_DIR,
    TIMESTAMP_FORMAT,
    feature_columns,
)

REPORT_MD = REPO_ROOT / "reports" / "data_audit.md"
REPORT_JSON = REPO_ROOT / "reports" / "data_audit.json"
MANIFEST = REPO_ROOT / "data" / "MANIFEST.md"

SEED = 42
SAMPLE_SIZE = 200_000
CORR_THRESHOLD = 0.99
NEAR_CONSTANT_SHARE = 0.99  # most frequent value covers >= 99% of rows
SCAN_PORTS = 500  # BENIGN (Src IP, Dst IP) pairs with more distinct Dst Ports are listed
WINDOW_GAP_S = 300  # a gap longer than 5 min between attack flows starts a new window
ORIGINAL_COLUMNS_N = 84

DAY_ABBR = {"Mon": "Monday", "Tue": "Tuesday", "Wed": "Wednesday", "Thu": "Thursday", "Fri": "Friday"}


# ----------------------------------------------------------------- helpers

def _fmt(n) -> str:
    if isinstance(n, (int, np.integer)):
        return f"{int(n):,}"
    if isinstance(n, (float, np.floating)):
        return f"{float(n):.4g}"
    return str(n)


def _pct(a, b) -> str:
    return "–" if not b else f"{100 * a / b:.2f}%"


def _md_table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(_fmt(x) for x in r) + " |")
    return "\n".join(out)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    return o


def parse_manifest_counts(path: Path) -> dict[str, int]:
    """Parse the label-count table in MANIFEST.md into {label: count}."""
    expected: dict[str, int] = {}
    text = path.read_text(encoding="utf-8")
    section = text.split("## Label counts", 1)[1].split("\n## ", 1)[0]
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 3 or cells[0] in ("Day", "") or set(cells[0]) <= set("-"):
            continue
        _, label, count = cells
        counts = [int(c.replace(",", "").strip()) for c in count.split("/")]
        if len(counts) == 2:
            base = re.sub(r"\s*/\s*[–-]\s*Attempted\s*$", "", label)
            expected[base] = counts[0]
            expected[f"{base} - Attempted"] = counts[1]
        else:
            expected[label] = counts[0]
    return expected


def _windows(seconds: np.ndarray, gap: int) -> list[tuple[int, int, int]]:
    """Split sorted epoch seconds into windows separated by gaps > ``gap`` s."""
    s = np.sort(seconds)
    breaks = np.flatnonzero(np.diff(s) > gap)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(s) - 1]
    return [(int(s[a]), int(s[b]), int(b - a + 1)) for a, b in zip(starts, ends)]


def _ts(sec: int) -> str:
    return pd.Timestamp(sec, unit="s").strftime("%Y-%m-%d %H:%M:%S")


def _hms(sec: int) -> str:
    return pd.Timestamp(sec, unit="s").strftime("%H:%M:%S")


# ----------------------------------------------------------------- main audit

def run_audit() -> dict:
    files = {d: PROCESSED_DIR / f"{d}.parquet" for d in DAYS}
    for d, p in files.items():
        if not p.exists():
            raise FileNotFoundError(f"{p} missing; run src/load_data.py first.")

    first = pd.read_parquet(files[DAYS[0]])
    all_cols = list(first.columns)
    original_cols = all_cols[:ORIGINAL_COLUMNS_N]
    feats = feature_columns(original_cols)
    assert len(original_cols) == 84 and len(feats) == 76, (len(original_cols), len(feats))
    del first

    n_total = 0
    day_rows = {}
    for d in DAYS:
        day_rows[d] = pq.ParquetFile(files[d]).metadata.num_rows
        n_total += day_rows[d]
    rng = np.random.default_rng(SEED)
    sample_global = np.sort(rng.choice(n_total, size=min(SAMPLE_SIZE, n_total), replace=False))

    res: dict = {
        "meta": {
            "seed": SEED,
            "sample_size": int(len(sample_global)),
            "corr_threshold": CORR_THRESHOLD,
            "near_constant_share": NEAR_CONSTANT_SHARE,
            "window_gap_seconds": WINDOW_GAP_S,
            "timestamp_format": TIMESTAMP_FORMAT,
            "n_features": len(feats),
            "float32_features": list(FLOAT32_FEATURES),
        },
        "files": {},
        "days": {},
    }

    label_counts_total: Counter = Counter()
    nan_counts = defaultdict(dict)
    posinf_counts = defaultdict(dict)
    neginf_counts = defaultdict(dict)
    neg_counts = defaultdict(dict)
    neg_labels = defaultdict(Counter)
    col_min = {c: np.inf for c in feats}
    col_max = {c: -np.inf for c in feats}
    value_counts: dict[str, pd.Series] = {}
    col_equal = {pair: True for pair in combinations(feats, 2)}

    hash_feat_parts, hash_featport_parts = [], []
    label_parts, day_parts, ts_parts = [], [], []
    sample_parts = []
    all_col_dups = {}
    flow_id_dup_rows = {}
    per_label_ids = {}
    per_label_time = {}
    exact_dup_feat_day = {}
    nonfinite_rows = {}
    scan_like = []
    offset = 0

    for d in DAYS:
        p = files[d]
        df = pd.read_parquet(p)
        n = len(df)
        res["files"][d] = {
            "parquet": str(p.relative_to(REPO_ROOT)),
            "parquet_bytes": p.stat().st_size,
            "csv_bytes": (RAW_DIR / f"{d}-WorkingHours.csv").stat().st_size
            if (RAW_DIR / f"{d}-WorkingHours.csv").exists()
            else None,
            "rows": n,
            "columns": df.shape[1],
        }
        lc = df[LABEL_COLUMN].value_counts()
        label_counts_total.update(lc.to_dict())

        # --- timestamps / weekday check
        ts = df["Timestamp"]
        weekdays = sorted(set(ts.dt.day_name()))
        res["days"][d] = {
            "rows": n,
            "label_counts": {k: int(v) for k, v in lc.items()},
            "timestamp_min": ts.min(),
            "timestamp_max": ts.max(),
            "weekday_of_timestamps": weekdays,
            "dates": sorted(set(ts.dt.strftime("%Y-%m-%d"))),
        }

        # --- NaN / inf / negatives / min-max / value counts / column equality
        X = {c: df[c].to_numpy(dtype=np.float64) for c in feats}
        for c in feats:
            v = X[c]
            nn = int(np.isnan(v).sum())
            pi = int(np.isposinf(v).sum())
            ni = int(np.isneginf(v).sum())
            if nn:
                nan_counts[c][d] = nn
            if pi:
                posinf_counts[c][d] = pi
            if ni:
                neginf_counts[c][d] = ni
            neg = v < 0  # -inf counts as negative too; NaN is not
            k = int(neg.sum())
            if k:
                neg_counts[c][d] = k
                neg_labels[c].update(df.loc[neg, LABEL_COLUMN].value_counts().to_dict())
            fin = v[np.isfinite(v)]
            if fin.size:
                col_min[c] = min(col_min[c], float(fin.min()))
                col_max[c] = max(col_max[c], float(fin.max()))
            vc = pd.Series(v).value_counts(dropna=False)
            value_counts[c] = vc if c not in value_counts else value_counts[c].add(vc, fill_value=0)
        for pair in col_equal:
            if col_equal[pair]:
                a, b = pair
                col_equal[pair] = bool(np.array_equal(X[a], X[b], equal_nan=True))
        del X

        # --- duplicate rows
        all_dup_extra = int(df[original_cols].duplicated(keep="first").sum())
        all_dup_rows = int(df[original_cols].duplicated(keep=False).sum())
        all_col_dups[d] = {"extra_copies": all_dup_extra, "rows_in_dup_groups": all_dup_rows}
        exact_dup_feat_day[d] = {
            "extra_copies": int(df[feats].duplicated(keep="first").sum()),
            "rows_in_dup_groups": int(df[feats].duplicated(keep=False).sum()),
        }
        fid_counts = df["Flow ID"].value_counts()
        flow_id_dup_rows[d] = {
            "distinct_flow_ids": int(len(fid_counts)),
            "rows_sharing_flow_id": int(fid_counts[fid_counts > 1].sum()),
        }

        # Hash each row's features for cross-day duplicate and conflict analysis.
        # Cast to float64 so equal values hash equally across float32/float64 columns,
        # and map -0.0 to 0.0 so the hash matches value equality.
        F = df[feats].astype("float64") + 0.0
        hf = pd.util.hash_pandas_object(F, index=False).to_numpy()
        exact_dup_feat_day[d]["hash_extra_copies"] = int(pd.Series(hf).duplicated().sum())
        exact_dup_feat_day[d]["hash_matches_exact"] = (
            exact_dup_feat_day[d]["hash_extra_copies"] == exact_dup_feat_day[d]["extra_copies"]
        )
        nonfinite_rows[d] = int((~np.isfinite(F.to_numpy())).any(axis=1).sum())
        hash_feat_parts.append(hf)
        F["Dst Port"] = df["Dst Port"].astype("float64")
        F["Protocol"] = df["Protocol"].astype("float64")
        hash_featport_parts.append(pd.util.hash_pandas_object(F, index=False).to_numpy())
        del F
        label_parts.append(df[LABEL_COLUMN].to_numpy(dtype=object))
        day_parts.append(np.full(n, d, dtype=object))
        ts_parts.append(ts.to_numpy().astype("datetime64[s]").astype(np.int64))

        # --- identifier diversity per label
        g = df.groupby(LABEL_COLUMN)
        ids = g.agg(
            rows=(LABEL_COLUMN, "size"),
            src_ip=("Src IP", "nunique"),
            dst_ip=("Dst IP", "nunique"),
            src_port=("Src Port", "nunique"),
            dst_port=("Dst Port", "nunique"),
            protocols=("Protocol", lambda s: ",".join(str(x) for x in sorted(s.unique()))),
        )
        for lab, row in ids.iterrows():
            sub = df.loc[df[LABEL_COLUMN] == lab]
            top_ports = sub["Dst Port"].value_counts().head(3)
            top_dst = sub["Dst IP"].value_counts().head(1)
            top_src = sub["Src IP"].value_counts().head(1)
            per_label_ids[(d, lab)] = {
                "day": d,
                "label": lab,
                **{k: (v if isinstance(v, str) else int(v)) for k, v in row.items()},
                "top_dst_ports": [[int(pt), int(c), round(c / len(sub), 4)] for pt, c in top_ports.items()],
                "top_dst_ip": [str(top_dst.index[0]), round(int(top_dst.iloc[0]) / len(sub), 4)],
                "top_src_ip": [str(top_src.index[0]), round(int(top_src.iloc[0]) / len(sub), 4)],
            }
            per_label_time[(d, lab)] = (sub["Timestamp"].min(), sub["Timestamp"].max(), len(sub))

        # --- scan-like BENIGN traffic: one (Src IP, Dst IP) pair touching many Dst Ports
        ben = df.loc[df[LABEL_COLUMN] == BENIGN]
        pairs = ben.groupby(["Src IP", "Dst IP"]).agg(
            rows=("Dst Port", "size"), dst_ports=("Dst Port", "nunique"),
            first=("Timestamp", "min"), last=("Timestamp", "max"))
        for (sip, dip), r in pairs.loc[pairs["dst_ports"] > SCAN_PORTS].iterrows():
            scan_like.append({"day": d, "src_ip": sip, "dst_ip": dip, "rows": int(r["rows"]),
                              "distinct_dst_ports": int(r["dst_ports"]),
                              "first": r["first"], "last": r["last"]})
        del ben, pairs

        # --- sample rows for correlation
        local = sample_global[(sample_global >= offset) & (sample_global < offset + n)] - offset
        sample_parts.append(df.iloc[local][feats].astype("float64"))
        offset += n
        del df, ts, g, ids
        gc.collect()
        print(f"[audit] processed {d}", flush=True)

    # ---------------------------------------------------------- label check vs manifest
    expected = parse_manifest_counts(MANIFEST)
    observed = dict(label_counts_total)
    labels_all = sorted(set(expected) | set(observed))
    label_check = [
        {"label": lab, "manifest": expected.get(lab), "observed": observed.get(lab, 0),
         "match": expected.get(lab) == observed.get(lab, 0)}
        for lab in labels_all
    ]
    manifest_rows = {
        "Monday": 371_749, "Tuesday": 322_003, "Wednesday": 496_779,
        "Thursday": 362_368, "Friday": 547_915,
    }
    res["label_check"] = {
        "all_match": all(r["match"] for r in label_check),
        "rows": label_check,
        "row_counts_match_manifest": all(res["days"][d]["rows"] == manifest_rows[d] for d in DAYS),
        "total_rows": n_total,
    }

    # ---------------------------------------------------------- NaN / inf / negatives
    res["nan_inf"] = {
        c: {
            "nan": sum(nan_counts[c].values()),
            "posinf": sum(posinf_counts[c].values()),
            "neginf": sum(neginf_counts[c].values()),
            "nan_by_day": nan_counts[c],
            "posinf_by_day": posinf_counts[c],
            "neginf_by_day": neginf_counts[c],
        }
        for c in feats
        if c in nan_counts or c in posinf_counts or c in neginf_counts
    }
    res["negatives"] = {
        c: {"count": sum(neg_counts[c].values()), "min": col_min[c], "by_day": neg_counts[c],
            "by_label": dict(neg_labels[c])}
        for c in feats if c in neg_counts
    }

    # ---------------------------------------------------------- constant / near-constant
    const, near = [], []
    for c in feats:
        vc = value_counts[c].sort_values(ascending=False)
        top_share = float(vc.iloc[0] / n_total)
        entry = {"column": c, "n_unique": int(len(vc)), "top_value": float(vc.index[0]),
                 "rows_other_values": int(n_total - vc.iloc[0]),
                 "top_share": top_share}
        if len(vc) == 1:
            const.append(entry)
        elif top_share >= NEAR_CONSTANT_SHARE:
            near.append(entry)
    res["constant_columns"] = const
    res["near_constant_columns"] = sorted(near, key=lambda e: -e["top_share"])
    del value_counts
    gc.collect()

    # ---------------------------------------------------------- duplicate columns
    const_names = {e["column"] for e in const}
    groups: list[list[str]] = []
    for (a, b), eq in col_equal.items():
        if not eq:
            continue
        for grp in groups:
            if a in grp or b in grp:
                grp.extend(x for x in (a, b) if x not in grp)
                break
        else:
            groups.append([a, b])
    res["identical_column_groups"] = [
        {"columns": grp, "all_constant": all(c in const_names for c in grp)} for grp in groups
    ]

    # ---------------------------------------------------------- correlations on sample
    S = pd.concat(sample_parts, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    del sample_parts
    nonconst = [c for c in feats if S[c].nunique(dropna=True) > 1]
    corr = S[nonconst].corr()
    identical_pairs = {frozenset(p) for p, eq in col_equal.items() if eq}
    corr_pairs = []
    for a, b in combinations(nonconst, 2):
        r = corr.at[a, b]
        if np.isfinite(r) and abs(r) > CORR_THRESHOLD:
            corr_pairs.append({"a": a, "b": b, "r": float(r),
                               "identical": frozenset((a, b)) in identical_pairs})
    corr_pairs.sort(key=lambda e: -abs(e["r"]))
    res["correlated_pairs"] = corr_pairs
    res["meta"]["corr_sample_constant_columns_skipped"] = [c for c in feats if c not in nonconst]
    del S, corr

    # Near-identical (not exactly equal) pairs: re-read only the needed columns.
    near_pairs = [(e["a"], e["b"]) for e in corr_pairs if not e["identical"] and e["r"] > 0.9999]
    need = sorted({c for p in near_pairs for c in p})
    stats = {p: {"rows_equal": 0, "max_abs_diff": 0.0, "max_rel_diff": 0.0} for p in near_pairs}
    if need:
        for d in DAYS:
            part = pd.read_parquet(files[d], columns=need)
            for a, b in near_pairs:
                x, y = part[a].to_numpy(np.float64), part[b].to_numpy(np.float64)
                eq = (x == y) | (np.isnan(x) & np.isnan(y))
                stats[(a, b)]["rows_equal"] += int(eq.sum())
                m = ~eq & np.isfinite(x) & np.isfinite(y)
                if m.any():
                    diff = np.abs(x[m] - y[m])
                    rel = diff / np.maximum(np.maximum(np.abs(x[m]), np.abs(y[m])), 1e-300)
                    stats[(a, b)]["max_abs_diff"] = max(stats[(a, b)]["max_abs_diff"], float(diff.max()))
                    stats[(a, b)]["max_rel_diff"] = max(stats[(a, b)]["max_rel_diff"], float(rel.max()))
                stats[(a, b)]["nonfinite_mismatch"] = stats[(a, b)].get("nonfinite_mismatch", 0) + int(
                    (~eq & ~(np.isfinite(x) & np.isfinite(y))).sum())
            del part
            gc.collect()
    res["near_identical_pairs"] = [
        {"a": a, "b": b, **stats[(a, b)], "share_rows_equal": stats[(a, b)]["rows_equal"] / n_total}
        for a, b in near_pairs
    ]

    # ---------------------------------------------------------- duplicate rows & label conflicts
    H = pd.DataFrame({
        "h": np.concatenate(hash_feat_parts),
        "hp": np.concatenate(hash_featport_parts),
        "label": np.concatenate(label_parts),
        "day": np.concatenate(day_parts),
        "ts": np.concatenate(ts_parts),
    })
    del hash_feat_parts, hash_featport_parts, label_parts, day_parts, ts_parts
    H["attack_type"] = H["label"].str.replace(r" - Attempted$", "", regex=True)
    H["is_benign"] = H["label"] == BENIGN
    H["is_attempted"] = H["label"].str.endswith(" - Attempted")

    def dup_summary(key: str) -> dict:
        dup_extra = int(H[key].duplicated(keep="first").sum())
        in_grp = H[key].duplicated(keep=False)
        g = H.groupby(key)
        n_labels = g["label"].transform("nunique")
        n_days = g["day"].transform("nunique")
        return {
            "extra_copies": dup_extra,
            "rows_in_dup_groups": int(in_grp.sum()),
            "distinct_vectors": int(H[key].nunique()),
            "rows_in_cross_day_groups": int((n_days > 1).sum()),
            "rows_in_conflicting_label_groups": int((n_labels > 1).sum()),
            "conflicting_groups": int((g["label"].nunique() > 1).sum()),
        }

    feat_dup = dup_summary("h")
    featport_dup = dup_summary("hp")

    # Per-label conflict breakdown (feature hash only).
    g = H.groupby("h")
    H["grp_nlabels"] = g["label"].transform("nunique")
    H["grp_has_benign"] = g["is_benign"].transform("any")
    H["_att"] = ~H["is_benign"] & ~H["is_attempted"]  # a real (non-Attempted) attack
    H["grp_has_attack"] = H.groupby("h")["_att"].transform("any")
    H["_type_nb"] = H["attack_type"].where(~H["is_benign"])  # NaN for BENIGN rows
    H["grp_n_attack_types"] = H.groupby("h")["_type_nb"].transform("nunique")
    H["grp_size"] = g["label"].transform("size")

    # Does the vector also occur with the row's "partner" label (X <-> "X - Attempted")?
    confl = H.loc[H["grp_nlabels"] > 1, ["h", "label", "attack_type", "is_attempted", "is_benign"]].copy()
    confl["partner"] = np.where(confl["is_attempted"], confl["attack_type"],
                                confl["attack_type"] + " - Attempted")
    confl.loc[confl["is_benign"], "partner"] = None
    present = confl[["h", "label"]].drop_duplicates().rename(columns={"label": "partner"})
    present["_hit"] = True
    confl = confl.merge(present, on=["h", "partner"], how="left")
    partner_hits = confl.groupby("label")["_hit"].apply(lambda s: int(s.fillna(False).astype(bool).sum()))
    del confl, present

    per_label_dup = []
    for lab, sub in H.groupby("label"):
        per_label_dup.append({
            "label": lab,
            "rows": int(len(sub)),
            "rows_in_dup_groups": int((sub["grp_size"] > 1).sum()),
            "extra_copies_within_label": int(sub["h"].duplicated(keep="first").sum()),
            "rows_conflicting_any_other_label": int((sub["grp_nlabels"] > 1).sum()),
            "rows_sharing_vector_with_BENIGN": int(sub["grp_has_benign"].sum()) if lab != BENIGN else None,
            "rows_sharing_vector_with_attack": int(sub["grp_has_attack"].sum()) if lab == BENIGN else None,
            "rows_sharing_vector_with_partner_label": int(partner_hits.get(lab, 0)) if lab != BENIGN else None,
            "rows_sharing_vector_with_other_attack_type": int((sub["grp_n_attack_types"] > 1).sum())
            if lab != BENIGN else None,
        })
    # Conflicting label pairs: which labels share vectors.
    conf = H.loc[H["grp_nlabels"] > 1, ["h", "label"]].drop_duplicates()
    pair_counter: Counter = Counter()
    for _, labs in conf.groupby("h")["label"]:
        for a, b in combinations(sorted(labs), 2):
            pair_counter[(a, b)] += 1
    conflict_pairs = [{"label_a": a, "label_b": b, "shared_vectors": int(k)}
                      for (a, b), k in pair_counter.most_common()]
    res["duplicates"] = {
        "all_columns_by_day": all_col_dups,
        "all_columns_total": {
            "extra_copies": sum(v["extra_copies"] for v in all_col_dups.values()),
            "rows_in_dup_groups": sum(v["rows_in_dup_groups"] for v in all_col_dups.values()),
        },
        "features_by_day_exact": exact_dup_feat_day,
        "features_overall_hash": feat_dup,
        "features_plus_dstport_protocol_overall_hash": featport_dup,
        "per_label": per_label_dup,
        "conflicting_label_pairs": conflict_pairs,
        "flow_id_reuse_by_day": flow_id_dup_rows,
        "rows_with_any_nan_or_inf_by_day": nonfinite_rows,
        "hash_note": "Overall/cross-day feature duplicates use a 64-bit row hash "
                     "(pandas hash_pandas_object on float64 values, -0.0 mapped to 0.0); "
                     "within-day counts also computed exactly with DataFrame.duplicated.",
    }

    # ---------------------------------------------------------- time structure
    time_by_label = []
    for (d, lab), (t0, t1, k) in sorted(per_label_time.items(), key=lambda kv: (DAYS.index(kv[0][0]), kv[1][0])):
        time_by_label.append({"day": d, "label": lab, "rows": k, "first": t0, "last": t1})
    windows_by_type = []
    day_summary = {}
    hourly = {}
    for d in DAYS:
        sub = H.loc[H["day"] == d]
        ben_ts = np.sort(sub.loc[sub["is_benign"], "ts"].to_numpy())
        attack_mask_total = np.zeros(len(ben_ts), dtype=bool)
        att_windows_day = []
        for t in sorted(sub.loc[~sub["is_benign"], "attack_type"].unique()):
            s_all = sub.loc[sub["attack_type"] == t, "ts"].to_numpy()
            s_att = sub.loc[(sub["attack_type"] == t) & ~sub["is_attempted"], "ts"].to_numpy()
            w = _windows(s_all, WINDOW_GAP_S)
            in_w = np.zeros(len(ben_ts), dtype=bool)
            for a, b, _ in w:
                lo, hi = np.searchsorted(ben_ts, a, "left"), np.searchsorted(ben_ts, b, "right")
                in_w[lo:hi] = True
            attack_mask_total |= in_w
            att_windows_day.extend(w)
            lo, hi = np.searchsorted(ben_ts, s_all.min(), "left"), np.searchsorted(ben_ts, s_all.max(), "right")
            largest = max(w, key=lambda x: x[2])
            windows_by_type.append({
                "day": d,
                "attack_type": t,
                "rows_incl_attempted": int(len(s_all)),
                "rows_attack_only": int(len(s_att)),
                "first": _ts(int(s_all.min())),
                "last": _ts(int(s_all.max())),
                "span_minutes": round((int(s_all.max()) - int(s_all.min())) / 60, 1),
                "n_windows": len(w),
                "covered_minutes": round(sum(b - a for a, b, _ in w) / 60, 1),
                "largest_window_share": round(largest[2] / len(s_all), 4),
                "max_internal_gap_minutes": round(float(np.diff(np.sort(s_all)).max()) / 60, 1) if len(s_all) > 1 else 0.0,
                "benign_rows_in_windows": int(in_w.sum()),
                "benign_rows_in_first_last_span": int(hi - lo),
                "benign_rows_day": int(len(ben_ts)),
                "windows": [{"start": _ts(a), "end": _ts(b), "rows": k} for a, b, k in w[:200]],
            })
        day_summary[d] = {
            "first": _ts(int(sub["ts"].min())),
            "last": _ts(int(sub["ts"].max())),
            "benign_rows": int(len(ben_ts)),
            "benign_rows_inside_any_attack_window": int(attack_mask_total.sum()),
            "benign_rows_outside_all_attack_windows": int((~attack_mask_total).sum()),
        }
        hrs = pd.to_datetime(sub["ts"], unit="s").dt.hour
        tab = pd.crosstab(hrs, sub["attack_type"])
        hourly[d] = {int(h): {k: int(v) for k, v in row.items() if v} for h, row in tab.iterrows()}
    res["time"] = {
        "by_label": time_by_label,
        "windows_by_attack_type": windows_by_type,
        "day_summary": day_summary,
        "hourly_counts": hourly,
    }

    res["scan_like_benign_pairs"] = scan_like

    # ---------------------------------------------------------- identifiers per label
    res["identifiers_by_label"] = [
        per_label_ids[k] for k in sorted(per_label_ids, key=lambda k: (DAYS.index(k[0]), k[1]))
    ]
    return res


# ----------------------------------------------------------------- markdown

def write_markdown(res: dict, path: Path) -> None:
    L = []
    m = res["meta"]
    L.append("# Data audit: corrected CICIDS2017 (WTMC 2021)\n")
    L.append(
        "Generated by `src/audit_data.py` from `data/processed/*.parquet` (written by "
        "`src/load_data.py`). Machine-readable copy: `reports/data_audit.json`. "
        "**This audit only reports.** Nothing has been dropped, imputed or selected; "
        "those are team decisions (protocol open decisions 1, 2 and 8).\n"
    )
    L.append(
        f"Settings: {m['n_features']} flow-feature columns (the 84 original columns minus Flow ID, "
        f"Src/Dst IP, Src/Dst Port, Protocol, Timestamp and Label); seed {m['seed']}; "
        f"correlation sample {m['sample_size']:,} rows; |r| > {m['corr_threshold']}; "
        f"near-constant means the most frequent value covers ≥ {m['near_constant_share']:.0%} of rows; "
        f"a new attack window starts after a gap of more than {m['window_gap_seconds'] // 60} min.\n"
    )

    # Files
    L.append("## 1. Files, rows and format\n")
    rows = []
    for d, f in res["files"].items():
        rows.append([d, f["rows"], f["columns"], f"{f['csv_bytes'] / 1e6:.1f} MB" if f["csv_bytes"] else "–",
                     f"{f['parquet_bytes'] / 1e6:.1f} MB", f["parquet_bytes"]])
    tot_pq = sum(f["parquet_bytes"] for f in res["files"].values())
    tot_csv = sum(f["csv_bytes"] or 0 for f in res["files"].values())
    rows.append(["**Total**", res["label_check"]["total_rows"], "", f"{tot_csv / 1e6:.1f} MB",
                 f"{tot_pq / 1e6:.1f} MB", tot_pq])
    L.append(_md_table(["Day", "Rows", "Columns", "CSV size", "Parquet size", "Parquet bytes"], rows))
    L.append(
        "\nThe Parquet files use zstd compression and contain the 84 original columns plus `day`, "
        "`attack_type`, `is_attempted` and `is_attack` (88 columns). Row counts match MANIFEST.md: "
        f"**{res['label_check']['row_counts_match_manifest']}**.\n"
    )
    L.append(
        f"- **Timestamp format:** `{m['timestamp_format']}`, for example `04/07/2017 01:53:44 PM`. "
        "This is day/month/year on a 12-hour clock with one-second resolution. Each file holds "
        "one date, and that date's weekday matches the file name:"
    )
    for d, dd in res["days"].items():
        L.append(f"  - {d}: date {', '.join(dd['dates'])} (weekday {', '.join(dd['weekday_of_timestamps'])}), "
                 f"{dd['timestamp_min']:%H:%M:%S} to {dd['timestamp_max']:%H:%M:%S}")
    L.append(
        "- No time zone is stated. The clock times (about 13:40 to 22:10) are about 5 h later than "
        "the published local schedule of the original CICIDS2017 (about 08:40 to 17:10, "
        "Atlantic Daylight Time). This is consistent with conversion to Central European Summer "
        "Time, but that is an inference; it is not documented. Timestamps are kept timezone-naive.\n"
        "- Text is parsed exactly: the parsed Timestamp formats back to the original string, and "
        "float values match Python's `float()`.\n"
        f"- **Dtypes:** {len(m['float32_features'])} integer-valued features are stored as float32 "
        "because the cast is exactly lossless on these files. The other 42 features are kept as "
        "float64: casting them to float32 would round microsecond durations and IATs above "
        "2^24 and would merge distinct mean/std/rate values (for example, Flow Duration would "
        "go from 867,994 to 853,382 distinct values). No feature would overflow float32. "
        "Ports are int32 and Protocol is int16. Details are in the `src/load_data.py` docstring.\n"
    )

    # Labels
    L.append("## 2. Label counts vs MANIFEST.md\n")
    rows = [[r["label"], r["manifest"] if r["manifest"] is not None else "–", r["observed"],
             "yes" if r["match"] else "**NO**"] for r in res["label_check"]["rows"]]
    L.append(_md_table(["Label", "MANIFEST", "Observed", "Match"], rows))
    L.append(f"\nAll label counts match: **{res['label_check']['all_match']}**.\n")
    rows = []
    for d, dd in res["days"].items():
        lc = dd["label_counts"]
        att = sum(v for k, v in lc.items() if k != BENIGN and not k.endswith(" - Attempted"))
        atm = sum(v for k, v in lc.items() if k.endswith(" - Attempted"))
        rows.append([d, dd["rows"], lc.get(BENIGN, 0), att, atm, _pct(att, dd["rows"])])
    L.append(_md_table(["Day", "Rows", "BENIGN", "Attack (is_attack)", "Attempted", "Attack share"], rows))
    L.append("")

    # NaN / inf
    L.append("## 3. NaN and ±inf values (only columns that have any)\n")
    rows = []
    for c, e in res["nan_inf"].items():
        by = "; ".join(f"{d[:3]} {e['nan_by_day'].get(d, 0)}/{e['posinf_by_day'].get(d, 0)}" for d in DAYS)
        rows.append([c, e["nan"], e["posinf"], e["neginf"], by])
    L.append(_md_table(["Column", "NaN", "+inf", "−inf", "By day (NaN/+inf)"], rows))
    nf = res["duplicates"]["rows_with_any_nan_or_inf_by_day"]
    L.append(f"\nRows with at least one NaN or ±inf feature: {sum(nf.values()):,} "
             f"({'; '.join(f'{d[:3]} {v:,}' for d, v in nf.items())}).\n")

    # Negatives
    L.append("## 4. Negative values in columns that should be non-negative\n")
    L.append("All 76 flow features are counts, lengths, durations, IATs, rates, flag counts, "
             "window sizes or bulk/active/idle statistics, so none should be negative.\n")
    if res["negatives"]:
        rows = []
        for c, e in res["negatives"].items():
            by = "; ".join(f"{d[:3]} {v}" for d, v in e["by_day"].items())
            labs = ", ".join(f"{k} {v}" for k, v in sorted(e["by_label"].items(), key=lambda kv: -kv[1])[:4])
            rows.append([c, e["count"], e["min"], by, labs])
        L.append(_md_table(["Column", "Negative rows", "Min", "By day", "By label (top 4)"], rows))
    else:
        L.append("No negative values in any feature column.")
    L.append("")

    # Constant columns
    L.append("## 5. Constant, near-constant and duplicate columns\n")
    L.append("### Constant columns (one unique value over all 2.1M rows)\n")
    L.append(_md_table(["Column", "Value"], [[e["column"], e["top_value"]] for e in res["constant_columns"]])
             if res["constant_columns"] else "None.")
    L.append(f"\n### Near-constant columns (most frequent value ≥ {m['near_constant_share']:.0%} of rows)\n")
    L.append(_md_table(["Column", "Unique values", "Most frequent value", "Share", "Rows with another value"],
                       [[e["column"], e["n_unique"], e["top_value"], f"{100 * e['top_share']:.3f}%",
                         e["rows_other_values"]]
                        for e in res["near_constant_columns"]]) if res["near_constant_columns"] else "None.")
    L.append("\n### Exactly identical columns (equal on every row; NaN = NaN)\n")
    rows = [[", ".join(g["columns"]), "yes" if g["all_constant"] else "no"] for g in res["identical_column_groups"]]
    L.append(_md_table(["Identical group", "All constant?"], rows) if rows else "None.")
    L.append("\n### Pairs with sample r > 0.9999 that are not exactly identical: row-by-row comparison on all rows\n")
    L.append("Pairs with a max relative difference of about 1e-13 are the same quantity up to "
             "floating-point rounding. Exact-equality checks (and exact-duplicate checks) treat them "
             "as different.\n")
    total_rows = res["label_check"]["total_rows"]
    rows = [[e["a"], e["b"], f"{100 * e['share_rows_equal']:.2f}%", total_rows - e["rows_equal"],
             e["max_abs_diff"], e["max_rel_diff"],
             e.get("nonfinite_mismatch", 0)] for e in res["near_identical_pairs"]]
    L.append(_md_table(["Column A", "Column B", "Rows exactly equal", "Rows differing", "Max |A−B|", "Max rel. diff",
                        "NaN/inf mismatches"], rows) if rows else "None.")
    L.append(
        f"\n### Highly correlated pairs (|r| > {m['corr_threshold']}, Pearson, seeded sample of "
        f"{m['sample_size']:,} rows; ±inf treated as missing; columns constant in the sample skipped)\n"
    )
    rows = [[e["a"], e["b"], f"{e['r']:.5f}", "yes" if e["identical"] else ""] for e in res["correlated_pairs"]]
    L.append(_md_table(["Column A", "Column B", "r", "Identical"], rows) if rows else "None.")
    L.append("")

    # Duplicates
    du = res["duplicates"]
    L.append("## 6. Duplicate rows and label conflicts\n")
    rows = []
    for d in DAYS:
        a, f = du["all_columns_by_day"][d], du["features_by_day_exact"][d]
        n = res["days"][d]["rows"]
        rows.append([d, n, a["extra_copies"], f["extra_copies"], _pct(f["extra_copies"], n), f["rows_in_dup_groups"]])
    fo = du["features_overall_hash"]
    rows.append(["**All days**", res["label_check"]["total_rows"], du["all_columns_total"]["extra_copies"],
                 fo["extra_copies"], _pct(fo["extra_copies"], res["label_check"]["total_rows"]),
                 fo["rows_in_dup_groups"]])
    L.append(_md_table(["Day", "Rows", "(a) Extra copies, all 84 columns",
                        "(b) Extra copies, 76 features", "(b) share", "(b) Rows in duplicate groups"], rows))
    fp = du["features_plus_dstport_protocol_overall_hash"]
    L.append(
        f"\n\"Extra copies\" is the number of rows that `drop_duplicates()` would remove. "
        "The per-day counts are exact (`DataFrame.duplicated`). The all-days figure for (b) uses a "
        "64-bit row hash and also counts duplicates across days. Within-day duplicates add up to "
        f"{sum(du['features_by_day_exact'][d]['extra_copies'] for d in DAYS):,}; the rest are vectors "
        f"repeated on another day. {fo['rows_in_cross_day_groups']:,} rows belong to a feature vector "
        f"that appears on more than one day. There are {fo['distinct_vectors']:,} distinct feature vectors.\n\n"
        f"If Dst Port and Protocol are added to the 76 features, there are {fp['extra_copies']:,} extra copies, "
        f"{fp['distinct_vectors']:,} distinct vectors, and {fp['rows_in_conflicting_label_groups']:,} rows in "
        f"label-conflicting groups ({fp['conflicting_groups']:,} groups).\n"
    )
    L.append(
        "Check: the per-day duplicate counts from the 64-bit hash equal the exact counts on every day: "
        f"**{all(du['features_by_day_exact'][d]['hash_matches_exact'] for d in DAYS)}**.\n"
    )
    L.append(
        f"**Label conflicts (76 features):** {fo['conflicting_groups']:,} distinct feature vectors carry "
        f"more than one `Label`. They cover {fo['rows_in_conflicting_label_groups']:,} rows.\n"
    )
    rows = []
    for e in du["per_label"]:
        rows.append([
            e["label"], e["rows"], e["rows_in_dup_groups"], e["extra_copies_within_label"],
            e["rows_conflicting_any_other_label"],
            e["rows_sharing_vector_with_attack"] if e["label"] == BENIGN else e["rows_sharing_vector_with_BENIGN"],
            e["rows_sharing_vector_with_partner_label"] if e["label"] != BENIGN else "–",
            e["rows_sharing_vector_with_other_attack_type"] if e["label"] != BENIGN else "–",
        ])
    L.append(_md_table(["Label", "Rows", "Rows in feature-dup groups", "Extra copies (same label)",
                        "Rows whose vector also has another label",
                        "…of which shared with BENIGN (for BENIGN: with a real attack)",
                        "…shared with its Attempted/non-Attempted partner label",
                        "…shared with a different attack type"], rows))
    L.append("\nLabel pairs that share identical feature vectors (top 25 by number of shared vectors):\n")
    rows = [[e["label_a"], e["label_b"], e["shared_vectors"]] for e in du["conflicting_label_pairs"][:25]]
    L.append(_md_table(["Label A", "Label B", "Shared vectors"], rows) if rows else "None.")
    L.append("\nFlow ID reuse (the same 5-tuple Flow ID on several rows, for example after flow timeouts):\n")
    rows = [[d, res["days"][d]["rows"], v["distinct_flow_ids"], v["rows_sharing_flow_id"]]
            for d, v in du["flow_id_reuse_by_day"].items()]
    L.append(_md_table(["Day", "Rows", "Distinct Flow IDs", "Rows whose Flow ID occurs more than once"], rows))
    L.append("")

    # Time
    t = res["time"]
    L.append("## 7. Time structure\n")
    L.append("### Time range per label\n")
    rows = [[e["day"], e["label"], e["rows"], f"{e['first']:%H:%M:%S}", f"{e['last']:%H:%M:%S}"]
            for e in t["by_label"]]
    L.append(_md_table(["Day", "Label", "Rows", "First", "Last"], rows))
    L.append(
        f"\n### Attack windows per attack type (Attempted rows included; a gap > {m['window_gap_seconds'] // 60} min "
        "starts a new window)\n"
    )
    rows = []
    for e in t["windows_by_attack_type"]:
        rows.append([e["day"][:3], e["attack_type"], e["rows_incl_attempted"], e["first"][11:], e["last"][11:],
                     e["span_minutes"], e["n_windows"], e["covered_minutes"],
                     f"{100 * e['largest_window_share']:.1f}%", e["max_internal_gap_minutes"],
                     e["benign_rows_in_windows"], _pct(e["benign_rows_in_windows"], e["benign_rows_day"])])
    L.append(_md_table(["Day", "Attack type", "Rows", "First", "Last", "Span (min)", "Windows", "Covered (min)",
                        "Largest window share", "Max gap (min)", "Benign rows in windows",
                        "% of day's benign"], rows))
    L.append("\nWindow lists (start to end, rows), up to 8 per type:\n")
    for e in t["windows_by_attack_type"]:
        ws = e["windows"][:8]
        more = f" … (+{e['n_windows'] - 8} more)" if e["n_windows"] > 8 else ""
        L.append(f"- **{e['attack_type']}** ({e['day'][:3]}): " +
                 "; ".join(f"{w['start'][11:]}–{w['end'][11:]} ({w['rows']:,})" for w in ws) + more)
    L.append("\n### Benign traffic inside vs outside attack windows, per day\n")
    rows = [[d, s["first"][11:], s["last"][11:], s["benign_rows"], s["benign_rows_inside_any_attack_window"],
             s["benign_rows_outside_all_attack_windows"]] for d, s in t["day_summary"].items()]
    L.append(_md_table(["Day", "First flow", "Last flow", "Benign rows", "Benign inside any attack window",
                        "Benign outside all attack windows"], rows))
    L.append("\n### Flows per clock hour (Timestamp hour; attack types include Attempted rows)\n")
    for d in DAYS:
        hc = t["hourly_counts"][d]
        types = sorted({k for v in hc.values() for k in v}, key=lambda k: (k != BENIGN, k))
        rows = [[f"{h:02d}:00"] + [hc[h].get(k, 0) for k in types] for h in sorted(hc)]
        L.append(f"**{d}**\n")
        L.append(_md_table(["Hour"] + types, rows))
        L.append("")

    # Identifiers
    L.append("## 8. Identifier diversity per label (for open decision 1: ports as features)\n")
    L.append("These are counts only, and no decision is made here. Top Dst Ports are shown as port (share of the label's rows).\n")
    rows = []
    for e in res["identifiers_by_label"]:
        tp = ", ".join(f"{p} ({100 * s:.1f}%)" for p, _, s in e["top_dst_ports"])
        rows.append([e["day"][:3], e["label"], e["rows"], e["src_ip"], e["dst_ip"], e["dst_port"], e["src_port"],
                     tp, f"{e['top_dst_ip'][0]} ({100 * e['top_dst_ip'][1]:.1f}%)", e["protocols"]])
    L.append(_md_table(["Day", "Label", "Rows", "Distinct Src IP", "Distinct Dst IP", "Distinct Dst Port",
                        "Distinct Src Port", "Top Dst Ports", "Top Dst IP", "Protocols"], rows))
    L.append("")
    L.append(f"## 9. Scan-like traffic labelled BENIGN (one Src IP to one Dst IP on more than {SCAN_PORTS} distinct Dst Ports)\n")
    L.append("This is reported only; it may be legitimate labelling, so check it against the WTMC 2021 "
             "documentation before acting on it.\n")
    rows = [[e["day"][:3], e["src_ip"], e["dst_ip"], e["rows"], e["distinct_dst_ports"],
             f"{e['first']:%H:%M:%S}", f"{e['last']:%H:%M:%S}"] for e in res["scan_like_benign_pairs"]]
    L.append(_md_table(["Day", "Src IP", "Dst IP", "Rows", "Distinct Dst Ports", "First", "Last"], rows)
             if rows else "None.")
    if rows:
        L.append(f"\nTotal: {sum(e['rows'] for e in res['scan_like_benign_pairs']):,} BENIGN rows.")
    L.append("")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    res = run_audit()
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(_jsonable(res), indent=1), encoding="utf-8")
    write_markdown(res, REPORT_MD)
    print(f"wrote {REPORT_MD.relative_to(REPO_ROOT)} and {REPORT_JSON.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
