# Utilities Overview

`utils/` contains support scripts that sit around the core benchmark runner.

## Runtime helpers

- `prompts.py`: benchmark prompts used by `main.py`
- `normalization.py`: text normalization utilities shared by evaluators
- `dataset_splits.py`: optional split loading and filtering helpers

## Report and reproduction utilities

- `generate_reports.py`: regenerate `results/report_*.json` from existing prediction files
- `generate_multirun_summary.py`: build precomputed summary tables for the dashboard multi-run tab
- `generate_multirun_appendix_latex.py`: export multi-run appendix tables for paper-style reporting

## Data preparation and maintenance

- `prep_labels.py`: labeling preparation helpers used during dataset construction
- `sync_to_gt.py`: maintenance script for syncing legacy label markdown into GT JSON

The scripts in the last group are not required for normal benchmark usage and are kept mainly for dataset maintenance or reproduction workflows.
