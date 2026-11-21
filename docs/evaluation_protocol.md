# Evaluation protocol

**Status: DRAFT, pending team sign-off.** Nothing below is final until the team approves it, including the open decisions at the end. Code: `src/evaluate.py`, tests: `tests/test_evaluate.py`.

## 1. Question and design

Can an Isolation Forest (IF) trained only on benign traffic detect attack types it has never seen better than a supervised Random Forest (RF) trained on benign traffic plus the known attack types? Data: the corrected CICIDS2017 flow dataset (Engelen, Rimmer & Joosen, WTMC 2021).

We use **leave-one-attack-type-out**. For each attack type *h* in the set of evaluated types *H*:

1. Hold out *h*, together with its "Attempted" variants, so neither model sees it in training or validation.
2. Train both models, calibrate each threshold on benign validation traffic, and freeze the thresholds.
3. Score the test set once. Report detection on *h* and the false-positive rate (FPR) on benign test traffic.

Repeat for seeds 42, 43 and 44.

## 2. Partitions

- **Train / validation / test** are split by day or by contiguous time block where possible, not by random row sampling, so that temporally adjacent near-duplicate flows don't leak across partitions. Split manifests are saved in `splits/`, and both models use the same manifests.
- **Duplicates are removed across partitions.** A feature vector (after feature selection) that appears in an earlier partition is dropped from later ones. Identical vectors with conflicting labels are logged.
- **Held-out type *h***: all its flows, including Attempted variants, are absent from train and validation. Its non-Attempted flows form the attack part of the test set. Attempted rows in test follow open decision 2.
- **Benign test** is the benign traffic in the test time block(s), and it is the same for every *h* where the split allows.
- **The test set is never used for any choice**: no features, hyperparameters, thresholds or early stopping.

## 3. What each model trains on

| Model | Training data | Score (higher = more attack-like) |
|---|---|---|
| Isolation Forest | Benign train rows only | `-model.score_samples(X)` |
| Random Forest | Benign train + known attack types (all types except *h*), binary label | `model.predict_proba(X)[:, 1]` |

The scikit-learn `predict()` and `contamination` settings are **not** used to make decisions. Both models use the threshold rule below.

## 4. Threshold rule (both models)

1. Score the benign validation rows. Set τ to the (1 − target_fpr) quantile of those scores, with target_fpr = 0.01.
2. Flag a flow as an attack iff **score > τ** (strictly greater).
3. Check that the validation FPR is ≤ target_fpr, then **freeze τ**.

Implementation (`calibrate_threshold`): with *n* benign validation scores, at most *k* = ⌊target_fpr · *n*⌋ false positives are allowed, and τ is the (*n* − *k*)-th smallest score. Ties at τ can only lower the achieved FPR. We use this conservative quantile instead of `np.quantile`'s default linear interpolation, which can exceed the target (for example, 2/150 = 1.33%). With fewer than 1/target_fpr = 100 benign validation rows, *k* = 0, so the code warns.

## 5. IF hyperparameter tuning

IF hyperparameters are tuned only against **known (non-held-out) attack types present in validation**, together with benign validation rows. The held-out type *h* and all test rows are never used. Candidates are compared by their mean detection rate on the known validation attacks at the calibrated 1% threshold. Tuning is redone inside each fold, because the set of "known" types changes with *h*.

## 6. Metrics

For held-out type *h*, model *m*, seed *s*, and frozen threshold τ:

- Detection rate: DR(*h*) = #{*x* ∈ A_h,test : score(*x*) > τ} / |A_h,test|
- False-positive rate: FPR(*h*) = #{*x* ∈ B_test : score(*x*) > τ} / |B_test|
- Macro detection per seed: Macro(*m*, *s*) = (1/|*H*|) Σ_h DR(*h*). This is unweighted, so every type counts equally regardless of size.
- Difference (percentage points): Δ = 100 · (mean_s Macro(IF, *s*) − mean_s Macro(RF, *s*))

**Success criterion:** Δ > 0 **and** every individual run of both models (every *h* and seed) has test FPR ≤ 1%.

## 7. Reporting

- Report percentages **and** raw counts (detected / total, false positives / benign total).
- Include a per-type table: mean DR with min–max across the 3 seeds, for IF and RF, plus Δ per type. Then a macro row (mean, and min–max of the per-seed macro).
- **Flag rare types as "uncertain"** when their test count is below the minimum (open decision 3). Keep them in the tables and in the macro average; don't drop them.
- **Always report the actual test FPR**, even when it is above 1%. If any run exceeds 1%, we **do not claim success**, and we say so plainly.
- Report Δ alongside the seed min–max. If the ranges overlap, describe the result as inconclusive rather than a win.

## 8. Open decisions for team sign-off

1. **Ports as features:** are `Src Port` / `Dst Port` allowed? They can let a model memorize attack-specific ports instead of learning behaviour. Proposal: drop `Src Port`, and discuss `Dst Port`.
2. **Attempted labels:** in **test**, should Attempted flows of *h* be (a) excluded, (b) counted as attacks, or (c) reported separately? Also, are Attempted flows of **known** types in train labelled attack, or dropped? Proposal: (a) for the primary metric, (c) as a secondary.
3. **Minimum test size per type:** below what count is a type "uncertain"? The code default of 50 is a placeholder. Should any type (e.g., Heartbleed, SQL Injection, Infiltration) be merged with a related type or excluded from *H*?
4. **Which types form *H*,** and whether to group variants (e.g., the four DoS tools, or the three web attacks) into one held-out family to reduce leakage between closely related types.
5. **Split boundaries:** the exact day/time blocks for train/validation/test, given that each attack type occurs on only one day. Decide how to guarantee benign validation and test data for every *h*.
6. **What the seeds vary:** model `random_state` only, or also any subsampling? Proposal: fixed splits, and the seeds vary models and subsampling.
7. **RF tuning:** use fixed defaults, or tune on known attacks in validation with the same rule as IF?
8. **Preprocessing:** how to handle infinite/NaN feature values and identifier columns (Flow ID, IPs, Timestamp). Proposal: drop identifiers, and fit any scaling on train only.
