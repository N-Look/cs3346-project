# CS3346 project context

This repository compares benign-only Isolation Forest with supervised Random
Forest on held-out attack types in corrected CICIDS2017 flow data.

- Start with [the README](README.md) for setup and the commands to regenerate the
  models, results, and figures. See [the paired comparison](reports/forest_comparison.md)
  for the current methodology and findings.
- Use the five corrected capture-day files described in [the data manifest](data/MANIFEST.md).
  Keep downloaded data, generated Parquet and split files, and CSV previews local.
- Preserve the same 73 ordered features, split manifests, and test rows for both
  models. Fit imputation on benign training rows, tune using known attacks in
  validation, and freeze the threshold before scoring the test set.
- The primary outputs are held-out attack recall and actual benign test
  false-positive rates. Report both rates and the test counts. Flag types with
  fewer than 50 test flows, and distinguish the 1% validation target from test
  performance.
- Treat the [team decision sheet](docs/team_decisions.md) as a draft. Document
  agreed methodology changes and rerun both models when changes affect their
  comparison.
- Update the [AI-use log](AI_USE_LOG.md) with the date, tool, purpose, files
  influenced, and what was verified. Record each group member's actual
  contribution.
- The [original work plan](docs/nicholas_agent_plan.md) is historical context.
  Use the README and comparison report for the implemented workflow.
