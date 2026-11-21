# Team decision sheet: features, data and splits

**Meeting:** ____________ (date) · **Present:** Fergus ☐ Ronit ☐ Arryan ☐ Nicholas ☐ William ☐
**Prepared by:** Nicholas · Evidence: `reports/data_audit.md`, `docs/evaluation_protocol.md`, `data/MANIFEST.md`

These choices must be fixed **before** anyone builds features, splits or models. Changing them later means rerunning everything. ★ marks the recommended option. Record the decision and move on; any changes after sign-off go in the log at the bottom.

---

### A. Features

| # | Question | Evidence | Options | Decision |
|---|---|---|---|---|
| A1 | Drop identifier columns (Flow ID, Src/Dst IP, Timestamp)? | They identify hosts and times rather than behaviour. | ★ Drop all · Keep some | |
| A2 | Use Src/Dst Port as features? | Every attack except Bot and PortScan uses **exactly 1** Dst Port, so the RF would learn "port = attack". | ★ Drop both · Keep Dst Port | |
| A3 | Protocol column? | Only 3 values. It describes behaviour and isn't an ID. | ★ Keep · Drop | |
| A4 | Redundant columns? | 3 pairs are the same quantity (e.g. Fwd Packet Length Mean = Fwd Segment Size Avg), and 20 pairs have \|r\| > 0.99. | ★ Drop one of each identical pair only · Also prune \|r\| > 0.99 | |
| A5 | Bad values: 793 rows with NaN/inf and ~3k negative values (e.g. negative durations). | Concentrated in the rate and IAT columns. | ★ Set inf and negative values to NaN, then median-impute fitted on train only · Drop those rows | |

### B. Labels and data cleaning

| # | Question | Evidence | Options | Decision |
|---|---|---|---|---|
| B1 | "Attempted" flows (9,144)? | The authors say never to treat them as a class. 1,288 of the 1,470 Bot-Attempted flows are identical to benign ones. | ★ Drop from train and validation; leave them out of the main metric and report them separately · Relabel as benign | |
| B2 | DoS Hulk (159k rows)? | The WTMC documentation recommends **not** using DoS Hulk as a malicious class. | ★ Exclude it and note this in the limitations · Keep it | |
| B3 | ~71k Thursday "BENIGN" flows from 192.168.10.8 that each hit ~1,000 ports on 11 hosts. | They look like a scan, probably by the host compromised in the Infiltration attack (still to be verified). | ★ Remove them from all benign data, so IF doesn't learn a scan as normal · Keep them | |
| B4 | Duplicates: 18% of rows repeat another row's features, and 282k of those repeats fall on different days. | Identical rows in train and test inflate results. | ★ Deduplicate on (features, label), then remove from validation and test any feature vector already in train · Do nothing | |
| B5 | PortScan: 76k of its 159k flows are identical to benign flows. | No model can separate these flows. | ★ Keep them, and explain the ~50% detection cap in the report · Remove them | |

### C. Experiment design

| # | Question | Evidence | Options | Decision |
|---|---|---|---|---|
| C1 | What is held out at a time? | Holding out one DoS tool while RF trains on three others isn't really "unseen". The proposal says "one attack *category* at a time". | ★ **Families:** Brute Force (FTP, SSH), DoS (GoldenEye, Slowhttptest, slowloris), Web Attacks (Brute Force, XSS, SQL Injection), Bot, DDoS, PortScan. Per-type rates are still reported · Individual types | |
| C2 | Rare types: Heartbleed (11), Infiltration (32). | Too small for reliable rates. | ★ Report them as "uncertain" and exclude them from the macro average · Exclude them completely | |
| C3 | How to split? | Each attack is in 1–7 short time windows on one day, and every day has 152k–239k attack-free benign flows. | ★ Within each day, contiguous time blocks of 60% train / 20% validation / 20% test, for benign traffic and for each known attack's window · Whole days | |
| C4 | Minimum test size before a type counts as "uncertain". | The code uses 50 as a placeholder. | ★ 100 flows · 50 | |
| C5 | Hyperparameter tuning. | Tuning IF but not RF would bias the comparison. | ★ Small, equal-sized grids for both, tuned on known families in validation · Defaults for both | |
| C6 | What the seeds (42, 43, 44) vary. | | ★ Fixed splits; seeds vary the models and any subsampling · Splits vary too | |
| C7 | Success criterion. | The code requires every run to be ≤ 1% test FPR. | ★ Every run ≤ 1% (stricter, easier to defend) · Average ≤ 1% | |

### D. Team logistics

| # | Item | Decision / owner |
|---|---|---|
| D1 | Team GitHub repo (owner, URL). Nicholas's local repo is pushed into it. | |
| D2 | Preprocessing ownership. Suggestion: Nicholas does features and splits (A, B4, C3); Arryan, Ronit and Fergus do cleaning rules and checks (A5, B1–B3) and review the splits. | |
| D3 | Shared interfaces: Parquet data; split manifests in `splits/`; every model exposes `fit(X)` and `score(X)` (higher = more attack-like); metrics from `src/evaluate.py`. | |
| D4 | Who checks B3 against the WTMC/CIC documentation? | |
| D5 | Next meeting date, and weekly meeting time (notes go in the repo). | |

---

**Sign-off:** all five members agree to the decisions above. ☐ F ☐ R ☐ A ☐ N ☐ W

**Change log (after sign-off):**
| Date | Decision changed | New choice | Why | Agreed by |
|---|---|---|---|---|
