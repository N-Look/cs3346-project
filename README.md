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
.venv/bin/python -m src.load_data
.venv/bin/python -m src.preprocess
.venv/bin/python run_iforest.py
.venv/bin/python run_random_forest.py
.venv/bin/python compare_forests.py
```

The first command converts the source CSVs into `data/processed/`. The second
saves one Parquet file per day in `data/cleaned/` with the 73 selected features
and invalid measurements marked missing. The model runner reads those files
and regenerates them if they are absent or the converted data changes. It reads
labels and times separately from the converted files and keeps the original
row order so the existing split manifests still apply. Missing values are
filled for each split using medians from its benign training rows.

Each model runs all eligible individual attack types at seeds 42, 43, and 44.
Isolation Forest saves the shared split manifests. Random Forest checks and
reuses them. The comparison command checks that feature lists, holdouts,
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
  `row_id` indexes the concatenated processed files from Monday to Friday,
  preserving each file's row order. See the
  [Isolation Forest methods](reports/iforest_notes.md) and
  [comparison report](reports/forest_comparison.md).

## Share with the group

Commit the two model runners, comparison script, `src/`, `tests/`, `docs/`,
`reports/`, the final comparison files in `results/`, `requirements.txt`,
`data/MANIFEST.md`, `AI_USE_LOG.md`, `.gitignore` and this README. Keep
`.venv/`, downloaded data, converted and cleaned Parquet files, CSV previews,
and generated split manifests local. The commands above rebuild them.

## Layout

- `src/` holds the loader, preprocessing, split builder, model and shared metrics.
- `tests/` covers data loading, split safeguards and evaluation.
- `docs/` holds the team's draft protocol, decision sheet, and original work plan.
- `reports/` holds the data audit and experiment write-up.
- `AGENTS.md` and `AI_USE_LOG.md` give AI collaborators the current project context
  and record AI-assisted work.
