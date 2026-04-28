# OCR Benchmark Framework

OCR Benchmark is a document understanding benchmark for vision-language models. The repository is designed for external researchers who want a clean, reproducible path for running OCR evaluation, comparing models, and reproducing benchmark summaries.

## 1. Repository Status

This repository is scoped around five stable modules:
- `runner`: benchmark execution from `main.py`
- `evaluators`: metrics and task-specific scoring logic
- `model adapters`: provider-specific wrappers in `models/`
- `dashboard`: Streamlit analysis and export UI
- `reproduction utilities`: report regeneration and multi-run summary scripts in `utils/`

Included in the open-source benchmark path:
- benchmark runner and model adapters
- evaluation code for `v1` and `v2`
- dashboard for report analysis
- summary generation scripts for multi-run experiments
- sample GT files and split definitions

Not part of the benchmark API surface:
- image datasets, which must be downloaded separately
- historical experiment caches under ignored output folders
- paper drafts and local research notes
- annotation workflow folders such as `labeling_v1/` and `labeling_v2/`

## 2. Tasks

The repository currently supports two benchmark tasks:

- `v1`: full-text OCR quality for document transcription
- `v2`: handwriting transcription plus Y/N option extraction

Single-run and multi-run use the same runner. Multi-run is simply the same benchmark flow with `--runs-per-image N`, where `N > 1`.

## 3. Repository Layout

Primary repository entry points:
- `main.py`: CLI benchmark runner
- `app.py`: Streamlit dashboard
- `models/`: model adapters for `openai`, `gemini`, `qwen`, `ollama`, and `dummy`
- `evaluators/`: metrics, evaluators, and statistical tests
- `dashboard/`: leaderboard, detailed view, statistics, export, and multi-run tabs
- `utils/`: report regeneration, multi-run summaries, and supporting scripts
- `data/`: GT JSON files and optional split definitions
- `tests/`: regression coverage for evaluation behavior

Optional or non-core local assets may still exist in the workspace, but they are not required for running the benchmark.

## 4. Setup

```bash
pip install -r requirements.txt
cp env.example .env
```

Fill in the API keys you need:
- `OPENAI_API_KEY`
- `GOOGLE_API_KEY`
- `DASHSCOPE_API_KEY`

Optional reproducibility setting:
- `OCR_BENCHMARK_SEED=42`

## 5. Data Preparation

Required GT files:
- `data/sample_gt_v1.json` or legacy-compatible `data/sample_gt.json`
- `data/sample_gt_v2.json`

Optional split file:
- `data/dataset_split.json`

Image files are not distributed with this repository. Place them in `data/` so that each image filename matches the corresponding `file_name` field in the GT JSON.

Current image source:
- Google Drive: [dataset images](https://drive.google.com/drive/folders/1yLLiAzUoAwD28IMYE2tZUibX_I-f5gXf?usp=drive_link)

## 6. Quickstart

Fastest smoke-test path:

```bash
python main.py -v v1 -m dummy -id dummy-smoke
python main.py -v v2 -m dummy -id dummy-smoke-v2
```

This generates:
- `results/preds_v1_dummy-smoke.json`
- `results/report_v1_dummy-smoke.json`
- `results/preds_v2_dummy-smoke-v2.json`
- `results/report_v2_dummy-smoke-v2.json`

## 7. Run The Benchmark

CLI interface:

```bash
python main.py -v {v1|v2} -m {provider} -id {model_id...} [--resume] [--no-postprocess] [--split PATH] [--runs-per-image N]
```

Key inputs:
- `-m`, `--model`: model provider (`dummy`, `gemini`, `qwen`, `openai`, `ollama`)
- `-id`, `--model_id`: one or more model IDs
- `-v`, `--version`: `v1` or `v2`
- `--gt`: custom GT JSON path
- `--split`: optional split JSON
- `--resume`: resume from existing prediction files
- `--no-postprocess`: disable evaluator post-processing
- `--runs-per-image N`: repeat each image `N` times

Examples:

```bash
# Single-run v1
python main.py -v v1 -m openai -id gpt-4.1-mini

# Single-run v2
python main.py -v v2 -m gemini -id gemini-2.0-flash-exp

# Ablation without evaluator post-processing
python main.py -v v2 -m openai -id gpt-4o --no-postprocess

# Multi-run benchmark with 3 runs per image
python main.py -v v1 -m openai -id gpt-4.1-mini --runs-per-image 3
```

Behavior notes:
- failed samples are persisted with `failed=true` and counted in report-level failure stats
- transient network errors are skipped instead of persisted, so `--resume` can retry them later
- when `--runs-per-image 1`, the run is the standard single-run benchmark
- when `--runs-per-image > 1`, the first run is also copied to the legacy single-run output path for dashboard compatibility

## 8. Result File Contract

Standard benchmark outputs:
- `results/preds_{version}_{model}.json`: primary prediction file used by the dashboard
- `results/report_{version}_{model}.json`: evaluation summary report

Multi-run outputs:
- `results/multirun/preds_{version}_{model}__run{n}.json`: raw predictions for each run
- `results/multirun/meta_{version}_{model}.json`: multi-run metadata and runtime environment snapshot

Precomputed multi-run summary artifacts:
- `results/multirun/per_run_{version}.json`
- `results/multirun/leaderboard_{version}.json`
- `results/multirun/leaderboard_std_{version}.json`
- `results/multirun/distribution_{version}.json`
- `results/multirun/summary_meta_{version}.json`

These files are generated artifacts and should usually stay out of git.

## 9. Multi-run Summary Generation

Generate precomputed artifacts for the dashboard multi-run tab:

```bash
python utils/generate_multirun_summary.py --version all --write-csv
```

You can also scope it per task:

```bash
python utils/generate_multirun_summary.py --version v1 --write-csv
python utils/generate_multirun_summary.py --version v2 --write-csv
```

This script consumes `results/multirun/preds_*__runN.json` and writes normalized summary files back into `results/multirun/`.

## 10. Regenerate Reports

If you already have prediction files and want to rebuild reports:

```bash
python utils/generate_reports.py
python utils/generate_reports.py --version v1
python utils/generate_reports.py --version v2 --no-postprocess
```

## 11. Dashboard

Launch the dashboard:

```bash
streamlit run app.py
```

Dashboard data dependencies:
- leaderboard, detailed view, statistics, and export tabs read `results/preds_*` and `results/report_*`
- the multi-run tab reads precomputed summary artifacts from `results/multirun/`
- optional paper-style figure views are hidden unless the required multi-run summary artifacts already exist

## 12. Metrics Summary

`v1` metrics:
- CER, WER, NED
- Precision, Recall, BoW F1, Exact Match

`v2` metrics:
- Weighted Score
- Y/N Accuracy
- Y-positive Precision, Recall, F1, Balanced Accuracy
- Handwriting CER, WER, NED

## 13. Testing

Example test command:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

Recommended smoke-test sequence before release:
- run a `dummy` single-run job for `v1`
- run a `dummy` single-run job for `v2`
- run a `dummy` multi-run job with `--runs-per-image 3`
- generate multi-run summaries
- launch the Streamlit dashboard

## 14. Limitations And Notes

- Image assets are external to the repository
- The repository may contain local ignored folders for paper drafting or dataset maintenance, but the benchmark does not depend on them
- Some maintenance scripts in `utils/` are intended for reproduction or dataset curation rather than normal benchmark usage
- The dashboard assumes benchmark outputs follow the filename conventions listed above
