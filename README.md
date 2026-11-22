# Detecting Unseen Network Attacks Using Anomaly Detection (CS3346A, Group 31)

Isolation Forest (benign-only) vs. Random Forest baseline on held-out CICIDS2017 attack types,
compared at a validation-calibrated 1% false-positive target.

## Setup
```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest
```
Raw data goes in `data/raw/` (not tracked by Git). See `data/MANIFEST.md` for sources and checksums.

## Run both models

Put the five corrected day CSVs under `data/raw/wtmc2021/`, with the names in
[the data manifest](data/MANIFEST.md). Run from this directory:

```bash
.venv/bin/python run_iforest.py
.venv/bin/python run_random_forest.py
.venv/bin/python compare_forests.py
```

Each model reads the original daily CSVs, selects the 73 features, and cleans
invalid measurements in memory. It preserves source row order and fills
missing values using medians from each fold's benign training rows.
Both runners generate their splits from the input and use seeds 42, 43,
and 44. Random Forest checks the saved split when one is present.
The comparison command checks that feature lists, holdouts,
seeds, and test counts match before generating the table and figure.

For a quick comparison, write both pilot runs to separate folders so the full
run's CSVs stay available:

```bash
.venv/bin/python run_iforest.py --holdout Bot --seed 42 \
  --output-dir results/pilot --split-dir splits/pilot
.venv/bin/python run_random_forest.py --holdout Bot --seed 42 \
  --output-dir results/pilot --split-dir splits/pilot
.venv/bin/python compare_forests.py --results-dir results/pilot
```

- `results/iforest_results.csv` has recall, test and validation false-positive
  rates, raw counts, thresholds, candidate settings, and timing per run.
- `results/iforest_summary.csv` gives per-type and unweighted macro results.
- `results/random_forest_results.csv` and `results/random_forest_summary.csv`
  give the baseline's per-run and summary metrics. Each model also writes its
  feature list in column order.
- `results/forest_comparison.csv` pairs the metrics, and
  `results/forest_recall.png` plots held-out recall with the range over seeds.
- `splits/iforest/` contains generated per-type Parquet manifests. Each
  `row_id` indexes the concatenated daily records from Monday to Friday,
  preserving each file's row order. See the
  [Isolation Forest methods](reports/iforest_notes.md) and
  [comparison report](reports/forest_comparison.md).

## Share with the group

Commit the two model runners, comparison script, `src/`, `tests/`, `docs/`,
`reports/`, the final comparison files in `results/`, `requirements.txt`,
`data/MANIFEST.md`, `AI_USE_LOG.md`, `.gitignore` and this README. Keep
`.venv/`, downloaded data, converted and cleaned Parquet files, CSV previews,
and generated split manifests local. Each model computes the features and splits it needs from the source CSVs.

## Layout

- `src/` holds the loader, preprocessing, split builder, model and shared metrics.
- `tests/` covers data loading, split safeguards and evaluation.
- `docs/` holds the team's draft protocol, decision sheet, and original work plan.
- `reports/` holds the data audit and experiment write-up.
- `AGENTS.md` and `AI_USE_LOG.md` give AI collaborators the current project context
  and record AI-assisted work.
