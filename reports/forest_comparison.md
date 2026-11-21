# Isolation Forest and Random Forest comparison

## Random Forest method

The supervised Random Forest baseline uses the same 73 cleaned flow features,
median imputation, and saved holdout manifests as Isolation Forest. Each fold
reserves one attack type for testing. Both models use the same seeded sample of
up to 10,000 benign training flows per capture day. Random Forest also trains on
all eligible known-attack training flows, with binary benign and attack labels.

Each candidate has 100 trees, samples up to 12,000 training flows with
replacement per tree, and weights classes inversely to their frequencies in
each tree's bootstrap sample. The three candidates use maximum depths of 12, 20,
and unrestricted. Candidates are ranked by mean recall over known validation
attack types at the 1% benign validation false-positive target. Both models use
the same tuning samples and objective. Ties go to the first candidate in the
listed order.

For the selected Random Forest, the attack score is the model's predicted
probability of the attack class. Its decision threshold is recalibrated on all
eligible benign validation rows and frozen before scoring test data. The
comparison uses every held-out attack row and the same 330,516 benign test
flows per run for both models. Test rates can differ from the validation target.

[Isolation Forest method and dataset details](iforest_notes.md) ·
[Reproduction instructions](../README.md)

## Results

The two models were evaluated on 13 attack types at three seeds. Random Forest
achieved 49.53% macro recall, compared with
26.77% for Isolation Forest. That is a
22.76 percentage-point advantage for Random Forest under the shared
validation target.

| Model | Macro recall | Range across seeds | Benign test false-positive rate | Runs within 1% |
|---|---:|---:|---:|---:|
| Isolation Forest | 26.77% | 21.91%–31.87% | 0.86%–0.97% | 39/39 |
| Random Forest | 49.53% | 40.64%–56.03% | 1.60%–3.84% | 0/39 |

Random Forest detected 99.84% of DDoS and
88.21% of PortScan flows on average, compared
with 0.02% and
0.00% for Isolation Forest. Both models
missed all FTP-Patator flows. Isolation Forest detected all 11 Heartbleed test
flows across seeds and Random Forest detected zero, with limited evidence from
that small test set. Across the nine attack types with at least 50 test flows,
macro recall was 15.09% for Isolation Forest and 53.07% for
Random Forest.

The proposal's success criterion was unmet. Isolation Forest stayed under the
1% test cap in every run and had lower average recall. Random Forest had higher
recall and exceeded the test cap in every run, despite meeting the validation
target. The recall gap therefore describes performance under a shared
calibration procedure with different observed test false-positive rates. The
time-based split, small samples for some types, and use of one capture dataset
limit how broadly the results generalize.

[Paired results by attack type](../results/forest_comparison.csv) ·
[Recall figure](../results/forest_recall.png) ·
[Random Forest per-run results](../results/random_forest_results.csv)
