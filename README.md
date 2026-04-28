# OCR Benchmark

OCR benchmark framework for document understanding with reproducible single-run and multi-run evaluation.

This repository is organized around five stable modules:
- runner
- evaluators
- model adapters
- dashboard
- reproduction utilities

What you can do here:
- run `v1` text-transcription benchmarking
- run `v2` handwriting + Y/N extraction benchmarking
- compare models with single-run or multi-run outputs
- inspect reports in the Streamlit dashboard

Start here:
- Main documentation: [README.en.md](README.en.md)
- 中文文档: [README.zh.md](README.zh.md)

Repository status:
- The benchmark code path lives in `main.py`, `app.py`, `models/`, `evaluators/`, `dashboard/`, `utils/`, `data/`, and `tests/`
- Image datasets are not distributed with the repository
- Research drafts, historical outputs, and local annotation assets are intentionally outside the benchmark API surface
