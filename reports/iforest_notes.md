# Isolation Forest methods and reproducibility

## Data and preprocessing

The experiment used the corrected CICIDS2017 flow dataset from Engelen, Rimmer and Joosen (2021). The starting files contained 2,100,814 labelled flows across five capture days. The source, raw file names, row counts and SHA-256 checksums are recorded in the [data manifest](../data/MANIFEST.md). [Corrected dataset and citation](https://intrusion-detection.distrinet-research.be/WTMC2021/tools_datasets.html)

Each run used 73 numeric flow features. Identifiers, ports, protocol and label metadata were excluded from the model inputs. Three redundant packet-length averages were removed. Infinite values and negative durations, rates and inter-arrival times in the affected fields were treated as missing. The imputer learned each column's median from that fold's eligible benign training rows. The workflow uses float32 model inputs and requires no scaling. [Data audit](data_audit.md) · [Feature list](../results/iforest_features.json)

The shared loader cleans the five daily source CSVs in memory on each run. Row eligibility and median imputation are applied for each held-out attack type.

## Experimental design

One individual attack type was held out per fold. The dataset authors recommend excluding DoS Hulk from the malicious classes for flow-based evaluation, so its rows were excluded. Flows marked `Attempted` were excluded from the primary dataset. Benign labels were otherwise kept as supplied. These settings and the rare-type rule below are working choices for the local runs, pending agreement on the shared comparison protocol. [Dataset documentation](https://intrusion-detection.distrinet-research.be/WTMC2021/extended_doc.html) · [Team decision sheet](../docs/team_decisions.md)

Within each day and label, eligible flows were sorted by start time and split into approximately 60% training, 20% validation and 20% test. Tied timestamps within a day and label were kept together. Every eligible flow of the held-out label was then reserved for test. Cleaned feature vectors were hashed before imputation to identify exact duplicates across partitions. Test rows took precedence over validation and training, and validation rows took precedence over training. This preserves the held-out attack and benign test denominators in every fold. Repeated flows within the same partition were retained.

## Isolation Forest and threshold calibration

Each Isolation Forest fit used a seeded sample of up to 10,000 benign training flows per capture day. Its anomaly score was the negative of scikit-learn's `score_samples` output. Candidate forests used 100 trees and tested three pairs of maximum samples per tree and fraction of features per tree: (256, 1.0), (1,024, 1.0) and (1,024, 0.75). For each candidate, the threshold was calibrated from up to 20,000 benign validation flows. The candidate with the highest mean recall across available known validation attack types was selected, using up to 2,000 validation flows per type. Ties went to the first pair in the listed order.

The selected model's decision threshold was recalibrated once on all eligible benign validation flows, then frozen. Only scores strictly above that threshold were flagged. This limits the validation false-positive rate to 1%. Test recall divides flagged held-out attack flows by all eligible flows of that type. The test false-positive rate divides flagged benign test flows by all benign test flows.

## Reproduction and baseline handoff

- The runs used Python 3.12.13 on macOS 26.5.1 (arm64), with scikit-learn 1.9.1, pandas 3.1.1 and NumPy 2.5.5. Full versions are in [requirements.txt](../requirements.txt).
- Seeds 42, 43 and 44 vary model fitting and training and tuning samples. Chronological splits are fixed across seeds. Follow the [README](../README.md) to regenerate the manifests and results.
- For Random Forest, load the same feature matrix with `load_experiment_inputs`, index it by the fold manifest's `row_id`, and fit the shared imputer on the manifest's benign training rows. The classifier can use both benign and known-attack training rows. Reuse the same benign validation rows for threshold calibration and the same benign and held-out attack test rows for reporting.
- This experiment measures within-dataset generalization. Chronological grouping and exact input matching reduce leakage. Related connections and near-duplicates can still span partitions, and removing cross-partition repeats changes the training distribution. The results also depend on the supplied benign labels and this capture environment.

## Isolation Forest results

Across 13 held-out types and three seeds (39 runs), the unweighted mean attack
recall was **26.77%**, ranging from 21.91% to 31.87% across
seeds. Every run stayed within the 1% benign test false-positive cap
(0.86%–0.97% observed). The false-positive denominator was 330,516
benign test flows per run.

Performance varied across attack types. Mean recall was 50.00% for DoS
Slowhttptest and 50.60% for DoS slowloris. Across all three seeds, the model
flagged zero PortScan, FTP-Patator or SQL injection flows. Heartbleed reached
100% recall across the three seeds on the same 11 test flows.

The primary macro recall includes all 13 eligible types. Heartbleed (11),
Infiltration (32), SQL injection (12) and XSS (27) are flagged as uncertain
because each has fewer than 50 test flows. As a sensitivity check, macro recall
was 15.09% across the nine types with at least 50 test flows. The difference
shows how strongly the overall average depends on a few small groups.

The [paired comparison](forest_comparison.md) reports the recall difference in
percentage points and the actual test false-positive rates for both models.

[Summary by attack type](../results/iforest_summary.csv) ·
[Per-run metrics and selected settings](../results/iforest_results.csv)
