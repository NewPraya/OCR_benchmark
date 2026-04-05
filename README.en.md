# OCR Benchmark Framework

OCR benchmark for vision-language models with reproducible evaluation and dashboard analysis.

## 1. What This Repo Does

- Runs OCR inference across multiple providers (`openai`, `gemini`, `qwen`, `ollama`, `dummy`)
- Evaluates results in two modes:
1. `v1`: full-text OCR metrics
2. `v2`: handwritten text + Y/N options (JSON output)
- Supports post-processing ablation (`--no-postprocess`)
- Provides statistical comparisons and Streamlit dashboard visualization

## 2. Repository Layout

- `main.py`: CLI runner for benchmark jobs
- `app.py`: Streamlit dashboard
- `models/`: model adapters
- `evaluators/`: metrics, v1/v2 evaluators, statistical tests
- `utils/`: prompts, dataset split utilities, labeling/report helpers
- `tests/`: regression tests for core evaluation behavior
- `data/`: input images and ground truth files
- `results/`: predictions and reports

## 3. Setup

```bash
pip install -r requirements.txt
cp env.example .env
```

Fill API keys in `.env`:
- `OPENAI_API_KEY`
- `GOOGLE_API_KEY`
- `DASHSCOPE_API_KEY` (for Qwen)

Optional reproducibility setting:
- `OCR_BENCHMARK_SEED` (default `42`, used in bootstrap CI sampling)

## 4. Data

Expected files:
- `data/sample_gt_v1.json` (or legacy `data/sample_gt.json`)
- `data/sample_gt_v2.json`
- Optional split file: `data/dataset_split.json`

Images are expected under `data/` and referenced by `file_name` in GT JSON.

## 5. Run Benchmark

```bash
# V1 text OCR
python main.py -v v1 -m openai -id gpt-4.1-mini

# V2 handwriting + Y/N
python main.py -v v2 -m gemini -id gemini-2.0-flash-exp

# Ablation (disable evaluator post-processing)
python main.py -v v2 -m openai -id gpt-4o --no-postprocess
```

Key behavior:
- Failed samples are persisted in prediction files (`failed=true`) and included in report-level failure stats.

## 6. Generate / Regenerate Reports

```bash
python utils/generate_reports.py
python utils/generate_reports.py --version v1
python utils/generate_reports.py --version v2 --no-postprocess
```

## 7. Dashboard

```bash
streamlit run app.py
```

Main tabs:
1. Leaderboard
2. Detailed View
3. Statistical Analysis
4. Export

## 8. Testing

Use your conda environment first:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate OCR_benchmark
python -m unittest discover -s tests -p 'test_*.py' -v
```

## 9. Metrics Summary

`v1`:
- CER, WER, NED (lower is better)
- Precision, Recall, BoW F1, Exact Match (higher is better)

`v2`:
- Weighted Score (higher is better)
- Y/N Accuracy (higher is better)
- Handwriting CER/WER/NED (lower is better)

## 10. Notes for Paper Submission

- Dependency versions are pinned in `requirements.txt`.
- Bootstrap CI uses a deterministic seed unless overridden by `OCR_BENCHMARK_SEED`.
- Deprecated schema-based evaluation modules were removed to keep the benchmark scope clear and reproducible.
