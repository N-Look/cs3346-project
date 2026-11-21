# Data manifest

## Corrected CICIDS2017 (WTMC 2021): primary dataset
- **Source:** https://intrusion-detection.distrinet-research.be/WTMC2021/Dataset/dataset.zip
  (linked from https://intrusion-detection.distrinet-research.be/WTMC2021/tools_datasets.html)
- **Downloaded:** 2026-11-20 by Nicholas
- **Citation:** G. Engelen, V. Rimmer, W. Joosen, "Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study," IEEE Security and Privacy Workshops (SPW), 2021, pp. 7–12. No licence stated on the source page.
- **Local path:** `data/raw/dataset.zip`, extracted to `data/raw/wtmc2021/` (not tracked by Git)

| File | Bytes | Rows (excl. header) | SHA-256 |
|------|------:|------:|---------|
| dataset.zip | 333,841,436 | – | 4d535da19795d85376ae1397d161329e3b06fc47d9a5a68cd9be2cd7ecee0f2a |
| Monday-WorkingHours.csv | 208,244,807 | 371,749 | 580bc5b3a48ccaa9b61d6936bb88adb00a6a069a02c974e946bd324bc09971bc |
| Tuesday-WorkingHours.csv | 178,453,210 | 322,003 | 59d60eff01526b511fa575a08a9a0e241674276adf1aa07471c947f7d733e19a |
| Wednesday-WorkingHours.csv | 291,529,516 | 496,779 | 820446eb19713031827a1893871e30fb9da4c3b21ad5cd15d9a2ef6686a37583 |
| Thursday-WorkingHours.csv | 187,707,525 | 362,368 | 3db32a8e9a4da80aa15fd0bbf69733c8de8c6e6147fe279770eec1be6c263d65 |
| Friday-WorkingHours.csv | 282,607,319 | 547,915 | 8bf5a79222d2a553b75fe1ee81cc3ddff89b87befaef52e62368fdb2e3a79def |

**Total:** 2,100,814 flows and 84 columns. The columns are 7 identifiers/metadata (Flow ID, Src/Dst IP, Src/Dst Port, Protocol, Timestamp), 76 flow features and `Label`.

## Label counts (`Label` column)

| Day | Label | Count |
|-----|-------|------:|
| All | BENIGN | 1,657,693 |
| Tue | FTP-Patator / – Attempted | 3,973 / 11 |
| Tue | SSH-Patator / – Attempted | 2,980 / 8 |
| Wed | DoS Hulk / – Attempted | 158,469 / 593 |
| Wed | DoS GoldenEye / – Attempted | 7,567 / 80 |
| Wed | DoS slowloris / – Attempted | 4,001 / 1,731 |
| Wed | DoS Slowhttptest / – Attempted | 1,742 / 3,369 |
| Wed | Heartbleed | 11 |
| Thu | Web Attack - Brute Force / – Attempted | 151 / 1,214 |
| Thu | Web Attack - XSS / – Attempted | 27 / 652 |
| Thu | Web Attack - Sql Injection | 12 |
| Thu | Infiltration / – Attempted | 32 / 16 |
| Fri | PortScan | 159,151 |
| Fri | DDoS | 95,123 |
| Fri | Bot / – Attempted | 738 / 1,470 |

Notes:
- Monday is benign only. Each attack type appears on exactly one day.
- Heartbleed (11), SQL Injection (12), XSS (27) and Infiltration (32) are too rare for reliable per-type rates. Mark them as uncertain, or merge them into families (team decision).
- The authors' WTMC documentation recommends **not** treating DoS Hulk as a malicious class. This is a team decision.
- The "– Attempted" labels must not be used as their own class (authors' guidance).

## Original CICIDS2017 (UNB): optional comparison, not downloaded
- https://www.unb.ca/cic/datasets/ids-2017.html
