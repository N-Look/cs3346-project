"""Tests for src/evaluate.py. Run with: .venv/bin/python -m pytest"""

import math
import warnings

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    MACRO_LABEL,
    calibrate_threshold,
    compare_models,
    detection_rate,
    evaluate_heldout,
    false_positive_rate,
    predict,
    summarize,
)


# ---------------------------------------------------------------- calibration

@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("n", [100, 150, 999, 10_000])
def test_threshold_respects_target_on_random_benign(seed, n):
    scores = np.random.default_rng(seed).normal(size=n)
    t = calibrate_threshold(scores, 0.01)
    fpr = false_positive_rate(scores, t)
    assert fpr <= 0.01
    # Not needlessly conservative with continuous scores: exactly floor(0.01 n) flagged.
    assert int(predict(scores, t).sum()) == math.floor(0.01 * n)


@pytest.mark.parametrize("target", [0.001, 0.01, 0.05, 0.07, 0.29, 0.5])
def test_threshold_respects_other_targets(target):
    scores = np.random.default_rng(0).uniform(size=1000)
    t = calibrate_threshold(scores, target)
    assert false_positive_rate(scores, t) <= target


def test_heavy_ties_respect_target():
    # 98% of scores identical at the top: flagging any tied value would give 98% FPR.
    scores = np.concatenate([np.zeros(20), np.full(980, 5.0)])
    t = calibrate_threshold(scores, 0.01)
    assert false_positive_rate(scores, t) <= 0.01


def test_ties_straddling_quantile_respect_target():
    # 1000 scores: 980 distinct low values, 15 tied at 2.0 (straddling the cut), 5 tied at 3.0.
    rng = np.random.default_rng(1)
    scores = np.concatenate([rng.uniform(0, 1, 980), np.full(15, 2.0), np.full(5, 3.0)])
    t = calibrate_threshold(scores, 0.01)  # k = 10 allowed, but 15 tied at 2.0
    fpr = false_positive_rate(scores, t)
    assert fpr <= 0.01
    assert fpr == pytest.approx(5 / 1000)  # only the 3.0 block is flagged


def test_all_identical_scores_give_zero_fpr():
    scores = np.ones(500)
    t = calibrate_threshold(scores, 0.01)
    assert false_positive_rate(scores, t) == 0.0


def test_beats_linear_quantile_counterexample():
    # np.quantile's linear method can flag 2/150 = 1.33% here; ours must not.
    scores = np.arange(150, dtype=float)
    t = calibrate_threshold(scores, 0.01)
    assert false_positive_rate(scores, t) <= 0.01


def test_small_sample_warns_and_flags_nothing():
    scores = np.arange(50, dtype=float)
    with pytest.warns(UserWarning):
        t = calibrate_threshold(scores, 0.01)
    assert t == 49.0
    assert false_positive_rate(scores, t) == 0.0


def test_no_warning_when_sample_large_enough():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        calibrate_threshold(np.arange(100, dtype=float), 0.01)


@pytest.mark.parametrize("bad_target", [0.0, 1.0, -0.1, 1.5, float("nan")])
def test_invalid_target_raises(bad_target):
    with pytest.raises(ValueError):
        calibrate_threshold(np.arange(100, dtype=float), bad_target)


@pytest.mark.parametrize("bad_scores", [[], np.array([]), [1.0, np.nan], [1.0, np.inf]])
def test_invalid_scores_raise(bad_scores):
    with pytest.raises(ValueError):
        calibrate_threshold(bad_scores, 0.01)


# ---------------------------------------------------------------- rates

def test_predict_is_strictly_greater():
    out = predict([0.0, 1.0, 1.0000001, 2.0], 1.0)
    assert out.dtype == bool
    assert out.tolist() == [False, False, True, True]


def test_perfect_separation_gives_full_detection_and_zero_fpr():
    benign_val = np.random.default_rng(0).uniform(0, 1, 1000)
    t = calibrate_threshold(benign_val, 0.01)
    attacks = np.random.default_rng(1).uniform(2, 3, 200)
    benign_test = np.random.default_rng(2).uniform(-1, 0, 1000)
    assert detection_rate(attacks, t) == 1.0
    assert false_positive_rate(benign_test, t) == 0.0


def test_rates_on_known_values():
    assert detection_rate([1, 2, 3, 4], 2) == 0.5
    assert false_positive_rate([1, 2, 3, 4, 5], 4) == 0.2


@pytest.mark.parametrize("fn", [detection_rate, false_positive_rate])
def test_rates_raise_on_empty_or_nan(fn):
    with pytest.raises(ValueError):
        fn([], 0.5)
    with pytest.raises(ValueError):
        fn([0.1, np.nan], 0.5)
    with pytest.raises(ValueError):
        fn([0.1, 0.2], float("nan"))


def test_evaluate_heldout_counts():
    r = evaluate_heldout(
        "IsolationForest", "PortScan", 42, 0.5,
        benign_test_scores=[0.1, 0.2, 0.6, 0.3],
        attack_test_scores=[0.9, 0.4, 0.7],
    )
    assert r["model"] == "IsolationForest" and r["heldout_type"] == "PortScan" and r["seed"] == 42
    assert (r["n_attack_test"], r["n_detected"]) == (3, 2)
    assert (r["n_benign_test"], r["n_false_positives"]) == (4, 1)
    assert r["detection_rate"] == pytest.approx(2 / 3)
    assert r["fpr"] == pytest.approx(0.25)


def test_evaluate_heldout_raises_on_empty_attacks():
    with pytest.raises(ValueError):
        evaluate_heldout("RF", "X", 42, 0.5, [0.1], [])


# ---------------------------------------------------------------- summarize / compare

def _row(model, htype, seed, n_att, n_det, n_ben, n_fp):
    return dict(
        model=model, heldout_type=htype, seed=seed, threshold=0.0,
        n_attack_test=n_att, n_detected=n_det, detection_rate=n_det / n_att,
        n_benign_test=n_ben, n_false_positives=n_fp, fpr=n_fp / n_ben,
    )


@pytest.fixture
def tiny_results():
    """2 models x 2 types x 2 seeds with easy numbers (benign n = 1000)."""
    rows = [
        # IF: type A detection 0.8 / 0.6, type B 0.4 / 0.2
        _row("IsolationForest", "A", 42, 100, 80, 1000, 10),
        _row("IsolationForest", "A", 43, 100, 60, 1000, 8),
        _row("IsolationForest", "B", 42, 10, 4, 1000, 10),
        _row("IsolationForest", "B", 43, 10, 2, 1000, 8),
        # RF: type A 0.5 / 0.5, type B 0.1 / 0.3
        _row("RandomForest", "A", 42, 100, 50, 1000, 5),
        _row("RandomForest", "A", 43, 100, 50, 1000, 6),
        _row("RandomForest", "B", 42, 10, 1, 1000, 5),
        _row("RandomForest", "B", 43, 10, 3, 1000, 6),
    ]
    return pd.DataFrame(rows)


def test_summarize_per_type_and_macro(tiny_results):
    s = summarize(tiny_results, min_attack_count=50).set_index(["model", "heldout_type"])
    a = s.loc[("IsolationForest", "A")]
    assert a["detection_mean"] == pytest.approx(0.7)
    assert (a["detection_min"], a["detection_max"]) == pytest.approx((0.6, 0.8))
    assert a["fpr_max"] == pytest.approx(0.010)
    assert a["n_detected"] == 140 and a["n_attack_test"] == 200
    assert not a["uncertain"]
    assert s.loc[("IsolationForest", "B"), "uncertain"]  # only 10 test flows

    m = s.loc[("IsolationForest", MACRO_LABEL)]
    # per-seed macro: seed 42 = (0.8+0.4)/2 = 0.6, seed 43 = (0.6+0.2)/2 = 0.4
    assert m["detection_mean"] == pytest.approx(0.5)
    assert (m["detection_min"], m["detection_max"]) == pytest.approx((0.4, 0.6))
    assert m["n_seeds"] == 2
    assert m["uncertain"]

    rm = s.loc[("RandomForest", MACRO_LABEL)]
    # seed 42 = (0.5+0.1)/2 = 0.3, seed 43 = (0.5+0.3)/2 = 0.4
    assert rm["detection_mean"] == pytest.approx(0.35)
    assert (rm["detection_min"], rm["detection_max"]) == pytest.approx((0.3, 0.4))


def test_compare_models_success(tiny_results):
    res = compare_models(summarize(tiny_results), target_fpr=0.01)
    assert res["macro_diff_pp"] == pytest.approx(15.0)  # 50% - 35%
    assert res["table"].loc["A", "diff_pp"] == pytest.approx(20.0)  # 70% - 50%
    assert res["table"].loc["B", "diff_pp"] == pytest.approx(10.0)  # 30% - 20%
    assert list(res["table"].index) == ["A", "B", MACRO_LABEL]
    assert res["worst_fpr_a"] == pytest.approx(0.010)
    assert res["worst_fpr_b"] == pytest.approx(0.006)
    assert res["fpr_ok"] and res["success"]


def test_compare_models_fails_when_any_run_exceeds_fpr(tiny_results):
    df = tiny_results.copy()
    idx = (df["model"] == "IsolationForest") & (df["heldout_type"] == "B") & (df["seed"] == 43)
    df.loc[idx, "n_false_positives"] = 11
    df.loc[idx, "fpr"] = 0.011
    res = compare_models(summarize(df), target_fpr=0.01)
    assert res["macro_diff_pp"] > 0
    assert res["worst_fpr_a"] == pytest.approx(0.011)
    assert not res["fpr_ok"] and not res["success"]


def test_compare_models_fails_when_a_not_better(tiny_results):
    res = compare_models(summarize(tiny_results), a="RandomForest", b="IsolationForest")
    assert res["macro_diff_pp"] == pytest.approx(-15.0)
    assert not res["success"]


def test_summarize_rejects_bad_input(tiny_results):
    with pytest.raises(ValueError):  # duplicate run
        summarize(pd.concat([tiny_results, tiny_results.iloc[[0]]]))
    with pytest.raises(ValueError):  # incomplete grid (IF type B missing seed 43)
        summarize(tiny_results.drop(index=3))
    with pytest.raises(ValueError):  # missing column
        summarize(tiny_results.drop(columns="fpr"))
    with pytest.raises(ValueError):
        summarize(tiny_results.iloc[0:0])


def test_compare_models_rejects_missing_model(tiny_results):
    s = summarize(tiny_results)
    with pytest.raises(ValueError):
        compare_models(s, a="IsolationForest", b="OneClassSVM")
    with pytest.raises(ValueError):  # different type coverage
        compare_models(s[~((s["model"] == "RandomForest") & (s["heldout_type"] == "B"))])
