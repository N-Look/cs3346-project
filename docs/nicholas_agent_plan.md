> Historical planning document. See the [current workflow](../README.md), [paired results](../reports/forest_comparison.md), and [AI-use log](../AI_USE_LOG.md) for the work actually completed. The milestones are internal.

# CS3346 Group 31: Nicholas's agent work plan

Scope: only Nicholas's parts of the approved proposal (`3346 Project Proposal.pdf`):
1. Data cleaning, feature selection, and preprocessing (shared with Arryan, Ronit, Fergus)
2. Baseline and evaluation-plan milestone (coordinator)
3. Isolation Forest model and tuning (shared with William)
4. Nicholas's sections of the final report and reproducibility appendix

Final deadline: **Mon Dec 7, 11:55 p.m. ET** (firm cutoff Wed Dec 9). Today: Fri Nov 20.

---

## Ground rules for every agent

- **AI use is documented.** The outline requires a written Brightspace confirmation of AI permission, and an AI-use statement giving the tool, date, purpose, material influenced and how it was verified. Every agent appends an entry to `AI_USE_LOG.md`.
- **Nicholas reviews before anything is shared.** Agents work on a local branch (`nicholas/<task>`). Nicholas reads, runs and understands each piece, then commits and pushes it himself. He has to be able to explain all of it, because the outline allows an individual project check (viva).
- **Don't touch teammates' work.** Agents only edit files in Nicholas's assigned areas, and they never push or open PRs.
- **The test set is off-limits for decisions.** No agent may use held-out-attack or test rows to pick features, settings or thresholds.
- **Reproducibility:** fixed seeds (42, 43, 44); package versions pinned in `requirements.txt`; raw data kept out of Git.

---

## Blockers to clear first (Nicholas, before any agent starts)

| # | What | Why |
|---|------|-----|
| B1 | **Disk space: only ~5 GB free (99% full).** Free about 15 GB, or run the data work in Google Colab, which the proposal allows. | The dataset CSVs are several GB, and the Python environment needs about 1 GB more. |
| B2 | Get the team GitHub repo URL and clone it locally (or confirm it doesn't exist yet and ask Fergus/Ronit to create it). | Agents need a place to put the code. |
| B3 | Agree with Arryan/Ronit/Fergus **which preprocessing pieces are yours.** Suggestion: you own the **feature list plus leakage checks** and the **split builder**. Others own per-day audits. | This stops four people (and agents) from rewriting the same code. |
| B4 | Agree the **shared interfaces** with the team: cleaned data format (Parquet), split manifest format, and the model API `fit(X) / score(X) -> higher = more attack-like`. | It lets the RF and IF work proceed in parallel. |
| B5 | Python setup: none of pandas, numpy or scikit-learn is installed (Python 3.14). Agent 0 will create a `uv` virtual environment. | Required before any code runs. |

---

## Agent work streams

```
Agent 0  Environment ─┐
Agent 1  Eval protocol ┼──────────────────────────────┐
Agent 2  Data + audit ─┴─> Agent 3 Features ─> Agent 4 Splits+pipeline ─> Agent 5 Isolation Forest ─> Agent 6 Report sections
                                                                                       └─> Agent 7 Independent reproduce check
```

Agents 0, 1 and 2 can run in parallel. Agents 3→4→5 run in sequence. Agents 6 and 7 run last.

### Agent 0: Environment setup (≈15 min, start now)
- Create `.venv` with `uv` on Python 3.12, a version scikit-learn supports well. Install pandas, numpy, scikit-learn, matplotlib, pyarrow and pytest. Write a pinned `requirements.txt`.
- **Output:** `requirements.txt`, plus a README "Setup" section.
- **Done when:** `python -c "import sklearn, pandas"` works in the venv.

### Agent 1: Evaluation protocol plus shared metrics code (milestone you coordinate; no data needed)
- Draft `docs/evaluation_protocol.md` covering:
  - leave-one-attack-type-out design
  - metrics: per-type detection rate, macro average, actual benign test false-positive rate (FPR), and percentage-point difference between IF and RF
  - threshold rule: the 99th percentile of benign-validation scores, frozen before testing
  - seeds
  - the **tuning rule for IF**: tune on known (non-held-out) attacks in validation, never on the held-out type
- Write `src/evaluate.py` with `calibrate_threshold(benign_val_scores, target_fpr=0.01)`, `detection_rate`, `false_positive_rate` and `summarize(...)`, plus `tests/test_evaluate.py`.
- **Output:** the protocol doc for the team to sign off at a recorded meeting; tested metric functions that William and Arryan's RF code can reuse.
- **Nicholas reviews:** whether the threshold rule matches the proposal, and whether the protocol is ready for team sign-off.

### Agent 2: Data download and audit (needs B1; asks before each download)
- Download the **corrected CICIDS2017 (Engelen, Rimmer & Joosen, WTMC 2021)** from the link in Section 4 of the proposal. Record file names, sizes, SHA-256 checksums and download date in `data/MANIFEST.md`. Add `data/raw/` to `.gitignore`.
- Audit each day:
  - row count, label counts (including "Attempted" variants)
  - NaN and inf counts per column
  - exact-duplicate rows; constant and duplicate columns
  - label-spelling inconsistencies
- Convert to Parquet with float32 to fit in 16 GB RAM.
- **Output:** `src/load_data.py`, `reports/data_audit.md` (tables), `data/MANIFEST.md`.
- **Nicholas reviews:** how Attempted labels are handled. Default: map to benign but keep the parent tag so splits can control for it. Also decide which attack types are too rare to test reliably.

### Agent 3: Feature selection and leakage checks (after Agent 2)
- Drop:
  - identifiers: Flow ID, Src/Dst IP, Timestamp
  - label columns
  - constant and duplicate columns
  - (team decision) Src/Dst Port. Flag this one, since ports can leak attack identity.
- Convert inf to NaN. Leave imputation for the pipeline.
- Leakage check: train a shallow tree on each single feature to predict "attack", and flag any feature that alone separates classes suspiciously well.
- **Output:** `config/features.yaml` (kept and dropped columns, with a reason for each), `reports/feature_selection.md`.
- **Nicholas reviews:** the port decision and the flagged features. Take these to the team.

### Agent 4: Split builder plus preprocessing pipeline (after Agent 3)
- For each attack type T, build a manifest:
  - **test** = all T flows + a benign test slice + known-attack test slice
  - **validation** = a benign validation slice + known-attack validation slice
  - **train** = benign train (+ known attacks, for RF only)
- Split by day or time block where possible. Remove duplicates across partitions. Keep T-Attempted flows out of train and validation.
- The pipeline (median imputer, plus an optional scaler) is **fit on train only**.
- Add tests for no row overlap between partitions, held-out type absent from train and validation, and the pipeline fit only on train rows.
- If runtime is too heavy, use the proposal's fallback: a seeded stratified benign subsample with documented caps. Test sets are left unchanged.
- **Output:** `src/splits.py`, `src/preprocess.py`, `splits/*.parquet` manifests, `tests/test_splits.py`.
- **Nicholas reviews:** split sizes per attack type. Share the manifests with the RF owners (William/Arryan) so both models use identical rows.

### Agent 5: Isolation Forest (after Agents 1 and 4; coordinate with William)
- `src/iforest.py` fits on benign train only. Score with `-score_samples` so higher means more anomalous. Calibrate the threshold with `evaluate.calibrate_threshold` on benign validation, then freeze it.
- Tune `n_estimators`, `max_samples` and `max_features` with a small grid, using known-attack validation detection at the frozen 1% threshold. Never use the held-out type.
- `run_experiments.py`: loop over every held-out type × seeds 42/43/44 and write `results/iforest_results.csv` (type, seed, threshold, detection rate, FPR, counts, runtime, params).
- Make one figure: per-type detection rate, IF vs RF (once RF results exist).
- **Output:** model code, results CSV, figure, `reports/iforest_notes.md` (what was tuned and why).
- **Nicholas reviews:** whether the test FPR is near 1%, which attack types IF misses and why, and whether the results make sense. Pair-review with William.

### Agent 6: Report drafts for Nicholas's sections (≈Dec 3–5)
- Draft from the actual outputs and logs:
  - Data and Preprocessing section
  - Isolation Forest Method section
  - Nicholas's parts of the reproducibility appendix (environment, data provenance and checksums, cleaning steps, IF settings and seeds, limitations)
  - an AI-use statement draft
- **Nicholas reviews:** rewrite it in his own voice and check every number against `results/`.

### Agent 7: Independent reproduce check (≈Dec 4–5)
- Fresh clone, fresh venv, follow the README only, and rerun one held-out type end to end. Report any step that fails or any number that differs.
- **Output:** `reports/repro_check.md`.

---

## Timeline

| Date | Agents | Nicholas |
|------|--------|----------|
| Nov 20–22 | 0, 1, 2 | Clear B1–B4. Hold and record the kickoff meeting. |
| Nov 23–26 | 3, 4 | Team signs off the evaluation protocol (milestone). Share split manifests with the RF owners. |
| Nov 27–Dec 1 | 5 | Pair-review IF with William; first held-out results. (Assignment 3 is due Nov 30.) |
| Dec 2–4 | 5 (final runs), 7 | Compare with RF. Fix anything the reproduce check found. |
| Dec 5–7 | 6 | Rewrite and finalize sections, then review the full report before Fergus submits. |

## Evidence to keep (participation rule)
- Your own commits for each piece you reviewed.
- Issues assigned to you.
- Meeting notes showing you attended.
- `AI_USE_LOG.md`.
