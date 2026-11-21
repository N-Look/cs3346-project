"""Shared evaluation code for the unseen-attack experiments.

Both models (Isolation Forest and Random Forest) are evaluated with exactly the
same functions so the comparison is fair. See ``docs/evaluation_protocol.md``.

Score convention
----------------
Every function here assumes **higher score = more attack-like**:

* Isolation Forest: ``score = -model.score_samples(X)``
* Random Forest:    ``score = model.predict_proba(X)[:, 1]`` (column 1 = attack)

A flow is flagged as an attack when ``score > threshold`` (strictly greater).

Threshold rule
--------------
The threshold is calibrated once on benign *validation* scores so that the
achieved validation false-positive rate is at most ``target_fpr``, then frozen.
Test data is never used to choose anything.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd

__all__ = [
    "calibrate_threshold",
    "predict",
    "detection_rate",
    "false_positive_rate",
    "evaluate_heldout",
    "summarize",
    "compare_models",
    "MACRO_LABEL",
]

#: Value used in the ``heldout_type`` column for the macro-average row.
MACRO_LABEL = "MACRO"

# Columns that ``summarize`` expects in its input (one row per model/type/seed).
_REQUIRED_RESULT_COLUMNS = (
    "model",
    "heldout_type",
    "seed",
    "n_attack_test",
    "n_detected",
    "detection_rate",
    "n_benign_test",
    "n_false_positives",
    "fpr",
)

# Tolerance for floating-point comparisons of rates (e.g. FPR <= target).
_EPS = 1e-12


def _as_scores(scores: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    """Convert ``scores`` to a 1-D float array, rejecting empty or non-finite input."""
    arr = np.asarray(scores, dtype=float)
    if arr.ndim != 1:
        arr = arr.ravel()
    if arr.size == 0:
        raise ValueError(f"{name} is empty; cannot compute a rate or threshold.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return arr


def _check_threshold(threshold: float) -> float:
    threshold = float(threshold)
    if math.isnan(threshold):
        raise ValueError("threshold is NaN.")
    return threshold


def calibrate_threshold(
    benign_val_scores: Sequence[float] | np.ndarray,
    target_fpr: float = 0.01,
) -> float:
    """Return a score threshold whose benign-validation FPR is at most ``target_fpr``.

    This is the (1 - target_fpr) quantile of the benign validation scores,
    computed conservatively so that ties can never push the achieved FPR over
    the target. Let ``n`` be the number of benign validation scores and
    ``k = floor(target_fpr * n)`` the largest number of false positives that is
    allowed. The threshold is the ``(n - k)``-th smallest score. Because flows
    are flagged only when ``score > threshold``, at most ``k`` benign
    validation flows are flagged (fewer if there are ties at the threshold).

    Note: ``np.quantile``'s default linear interpolation is *not* used, because
    it can exceed the target (e.g. n=150, target=0.01 can flag 2/150 = 1.33%).

    Args:
        benign_val_scores: Scores of benign validation flows (higher = more
            attack-like). Must be non-empty and finite.
        target_fpr: Desired maximum false-positive rate, strictly in (0, 1).

    Returns:
        The frozen threshold to use on the test set.

    Raises:
        ValueError: If the scores are empty / non-finite or ``target_fpr`` is
            not strictly between 0 and 1.

    Warns:
        UserWarning: If ``n < 1 / target_fpr``; then ``k = 0`` and the
            threshold is the maximum benign score (validation FPR = 0), which
            is a very coarse calibration.
    """
    target_fpr = float(target_fpr)
    if not (0.0 < target_fpr < 1.0) or math.isnan(target_fpr):
        raise ValueError(f"target_fpr must be strictly between 0 and 1, got {target_fpr}.")
    scores = np.sort(_as_scores(benign_val_scores, "benign_val_scores"))
    n = scores.size
    # Small tolerance so that e.g. 0.07 * 100 = 7.000000000000001 -> 7, and
    # 0.29 * 100 = 28.999999999999996 -> 29 (the intended integer).
    k = math.floor(target_fpr * n + 1e-9)
    if k == 0:
        warnings.warn(
            f"Only {n} benign validation scores for target_fpr={target_fpr}; "
            "threshold set to the maximum score (validation FPR = 0).",
            UserWarning,
            stacklevel=2,
        )
    k = min(k, n - 1)  # always keep at least one score at or below the threshold
    threshold = float(scores[n - k - 1])

    achieved = false_positive_rate(scores, threshold)
    if achieved > target_fpr + _EPS:  # should be impossible; guards against bugs
        raise RuntimeError(
            f"Calibration failed: validation FPR {achieved:.6f} > target {target_fpr}."
        )
    return threshold


def predict(scores: Sequence[float] | np.ndarray, threshold: float) -> np.ndarray:
    """Flag flows as attacks: returns a boolean array, ``True`` where ``score > threshold``.

    Args:
        scores: Attack scores (higher = more attack-like).
        threshold: Frozen threshold from :func:`calibrate_threshold`.

    Returns:
        Boolean NumPy array of the same length as ``scores``.
    """
    threshold = _check_threshold(threshold)
    arr = np.asarray(scores, dtype=float).ravel()
    if np.isnan(arr).any():
        raise ValueError("scores contain NaN values.")
    return arr > threshold


def detection_rate(attack_scores: Sequence[float] | np.ndarray, threshold: float) -> float:
    """Detection rate (recall) on attack flows: ``#{score > threshold} / #attacks``.

    Raises:
        ValueError: If ``attack_scores`` is empty or non-finite.
    """
    arr = _as_scores(attack_scores, "attack_scores")
    return float(np.mean(predict(arr, threshold)))


def false_positive_rate(benign_scores: Sequence[float] | np.ndarray, threshold: float) -> float:
    """False-positive rate on benign flows: ``#{score > threshold} / #benign``.

    Raises:
        ValueError: If ``benign_scores`` is empty or non-finite.
    """
    arr = _as_scores(benign_scores, "benign_scores")
    return float(np.mean(predict(arr, threshold)))


def evaluate_heldout(
    model_name: str,
    heldout_type: str,
    seed: int,
    threshold: float,
    benign_test_scores: Sequence[float] | np.ndarray,
    attack_test_scores: Sequence[float] | np.ndarray,
) -> dict[str, object]:
    """Evaluate one (model, held-out attack type, seed) run on the test set.

    Args:
        model_name: e.g. ``"IsolationForest"`` or ``"RandomForest"``.
        heldout_type: Name of the attack type that was held out of training.
        seed: Random seed of this run.
        threshold: Threshold frozen on benign validation scores.
        benign_test_scores: Scores of benign test flows.
        attack_test_scores: Scores of the held-out type's test flows only.

    Returns:
        A flat dict (one row of a results table) with keys ``model``,
        ``heldout_type``, ``seed``, ``threshold``, ``n_attack_test``,
        ``n_detected``, ``detection_rate``, ``n_benign_test``,
        ``n_false_positives`` and ``fpr``. Rates are fractions in [0, 1].
    """
    benign = _as_scores(benign_test_scores, "benign_test_scores")
    attack = _as_scores(attack_test_scores, "attack_test_scores")
    threshold = _check_threshold(threshold)
    n_detected = int(predict(attack, threshold).sum())
    n_fp = int(predict(benign, threshold).sum())
    return {
        "model": str(model_name),
        "heldout_type": str(heldout_type),
        "seed": int(seed),
        "threshold": threshold,
        "n_attack_test": int(attack.size),
        "n_detected": n_detected,
        "detection_rate": n_detected / attack.size,
        "n_benign_test": int(benign.size),
        "n_false_positives": n_fp,
        "fpr": n_fp / benign.size,
    }


def summarize(results_df: pd.DataFrame, min_attack_count: int = 50) -> pd.DataFrame:
    """Aggregate per-run results across seeds, per model and held-out type.

    Args:
        results_df: One row per (model, heldout_type, seed), e.g.
            ``pd.DataFrame([evaluate_heldout(...), ...])``.
        min_attack_count: Held-out types with fewer test flows than this (in
            any seed) are flagged ``uncertain=True``. They are kept, not
            dropped. The default of 50 is a placeholder pending team sign-off.

    Returns:
        DataFrame with one row per (model, heldout_type) plus one
        ``heldout_type == "MACRO"`` row per model, and columns:

        * ``n_seeds``
        * ``n_attack_test``, ``n_detected``, ``n_benign_test``,
          ``n_false_positives``: summed over seeds (raw counts for reporting)
        * ``detection_mean``, ``detection_min``, ``detection_max``
        * ``fpr_mean``, ``fpr_min``, ``fpr_max``
        * ``uncertain``

        For the MACRO row, the macro detection rate (and mean FPR) is first
        computed *within each seed* as the unweighted mean over held-out
        types; mean/min/max are then taken across seeds. Its ``fpr_max`` is
        therefore the max over seeds of the per-seed mean FPR, *not* the worst
        single run (use :func:`compare_models` for the worst-case check).
        Count columns in the MACRO row are sums over all types and seeds.

    Raises:
        ValueError: On missing columns, duplicate (model, type, seed) rows, or
            an incomplete grid (a model missing some type/seed combination).
    """
    missing = [c for c in _REQUIRED_RESULT_COLUMNS if c not in results_df.columns]
    if missing:
        raise ValueError(f"results_df is missing columns: {missing}")
    if results_df.empty:
        raise ValueError("results_df is empty.")
    keys = ["model", "heldout_type", "seed"]
    if results_df.duplicated(keys).any():
        raise ValueError("results_df has duplicate (model, heldout_type, seed) rows.")

    rows: list[dict[str, object]] = []
    count_cols = ["n_attack_test", "n_detected", "n_benign_test", "n_false_positives"]
    for model, mdf in results_df.groupby("model", sort=True):
        det = mdf.pivot(index="heldout_type", columns="seed", values="detection_rate")
        fpr = mdf.pivot(index="heldout_type", columns="seed", values="fpr")
        if det.isna().any().any():
            raise ValueError(
                f"Incomplete results for model {model!r}: every held-out type must "
                "have a result for every seed."
            )
        for htype, tdf in mdf.groupby("heldout_type", sort=True):
            row: dict[str, object] = {"model": model, "heldout_type": htype, "n_seeds": len(tdf)}
            row.update({c: int(tdf[c].sum()) for c in count_cols})
            row.update(
                detection_mean=float(tdf["detection_rate"].mean()),
                detection_min=float(tdf["detection_rate"].min()),
                detection_max=float(tdf["detection_rate"].max()),
                fpr_mean=float(tdf["fpr"].mean()),
                fpr_min=float(tdf["fpr"].min()),
                fpr_max=float(tdf["fpr"].max()),
                uncertain=bool(tdf["n_attack_test"].min() < min_attack_count),
            )
            rows.append(row)

        macro_det = det.mean(axis=0)  # per seed: unweighted mean over types
        macro_fpr = fpr.mean(axis=0)
        macro: dict[str, object] = {"model": model, "heldout_type": MACRO_LABEL, "n_seeds": det.shape[1]}
        macro.update({c: int(mdf[c].sum()) for c in count_cols})
        macro.update(
            detection_mean=float(macro_det.mean()),
            detection_min=float(macro_det.min()),
            detection_max=float(macro_det.max()),
            fpr_mean=float(macro_fpr.mean()),
            fpr_min=float(macro_fpr.min()),
            fpr_max=float(macro_fpr.max()),
            uncertain=bool((mdf["n_attack_test"] < min_attack_count).any()),
        )
        rows.append(macro)

    return pd.DataFrame(rows).reset_index(drop=True)


def compare_models(
    summary_df: pd.DataFrame,
    a: str = "IsolationForest",
    b: str = "RandomForest",
    target_fpr: float = 0.01,
) -> dict[str, object]:
    """Compare model ``a`` against model ``b`` using the output of :func:`summarize`.

    Success criterion (from the proposal): ``a``'s macro detection rate is
    strictly greater than ``b``'s, **and** every single run of both models
    (every held-out type and seed) has test FPR <= ``target_fpr``. The
    worst-case run FPR is the max of ``fpr_max`` over the per-type rows.

    Args:
        summary_df: Output of :func:`summarize` containing both models.
        a: Model hypothesised to be better (default Isolation Forest).
        b: Baseline model (default Random Forest).
        target_fpr: FPR ceiling for the success criterion.

    Returns:
        Dict with keys:

        * ``table``: DataFrame indexed by ``heldout_type`` (types then MACRO)
          with ``detection_mean_<a>``, ``detection_mean_<b>``,
          ``diff_pp`` (= 100 * (a - b), percentage points), the min/max of
          each model across seeds, and ``uncertain``.
        * ``macro_diff_pp``: macro detection difference in percentage points.
        * ``worst_fpr_a`` / ``worst_fpr_b``: highest test FPR of any run.
        * ``fpr_ok``: both worst-case FPRs <= ``target_fpr``.
        * ``success``: ``macro_diff_pp > 0 and fpr_ok``.

    Raises:
        ValueError: If either model is missing or they cover different
            held-out types.
    """
    out: dict[str, pd.DataFrame] = {}
    for name in (a, b):
        sub = summary_df[summary_df["model"] == name]
        if sub.empty:
            raise ValueError(f"Model {name!r} not found in summary_df.")
        out[name] = sub.set_index("heldout_type")
    types_a, types_b = set(out[a].index), set(out[b].index)
    if types_a != types_b:
        raise ValueError(
            f"Models cover different held-out types: only in {a}: {sorted(types_a - types_b)}, "
            f"only in {b}: {sorted(types_b - types_a)}"
        )

    order = sorted(t for t in types_a if t != MACRO_LABEL) + [MACRO_LABEL]
    ta, tb = out[a].loc[order], out[b].loc[order]
    table = pd.DataFrame(
        {
            f"detection_mean_{a}": ta["detection_mean"],
            f"detection_min_{a}": ta["detection_min"],
            f"detection_max_{a}": ta["detection_max"],
            f"detection_mean_{b}": tb["detection_mean"],
            f"detection_min_{b}": tb["detection_min"],
            f"detection_max_{b}": tb["detection_max"],
            "diff_pp": 100.0 * (ta["detection_mean"] - tb["detection_mean"]),
            "uncertain": ta["uncertain"] | tb["uncertain"],
        },
        index=pd.Index(order, name="heldout_type"),
    )

    per_type = [t for t in order if t != MACRO_LABEL]
    worst_a = float(ta.loc[per_type, "fpr_max"].max())
    worst_b = float(tb.loc[per_type, "fpr_max"].max())
    macro_diff = float(table.loc[MACRO_LABEL, "diff_pp"])
    fpr_ok = bool(worst_a <= target_fpr + _EPS and worst_b <= target_fpr + _EPS)
    return {
        "table": table,
        "macro_diff_pp": macro_diff,
        "worst_fpr_a": worst_a,
        "worst_fpr_b": worst_b,
        "fpr_ok": fpr_ok,
        "success": bool(macro_diff > 0 and fpr_ok),
    }
