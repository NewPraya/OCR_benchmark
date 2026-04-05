# PDF to PNG Conversion Guide

This guide explains how to convert PDF files in `data/prepare/` to PNG images and rename outputs with sequential numbers.

## Files

- `convert_pdf_to_png.py`: main conversion script
- `requirements_pdf_convert.txt`: Python dependencies for PDF conversion
- `setup_and_convert.sh`: one-command setup + conversion

## Quick Start

### Option 1: One-command script (recommended)

```bash
cd utils/pdfConverter
bash setup_and_convert.sh
```

The script will:
1. Activate conda env `OCR_benchmark`
2. Install poppler (system dependency)
3. Install Python dependencies
4. Run conversion

### Option 2: Manual steps

```bash
cd utils/pdfConverter
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate OCR_benchmark
conda install -c conda-forge poppler -y
pip install -r requirements_pdf_convert.txt
python convert_pdf_to_png.py
```

## Output

Converted PNG files are written to `data/prepare/png_output/`.

## Notes

- Default conversion quality: 300 DPI
- Default behavior: convert the first page of each PDF
- Output format: PNG

## Troubleshooting

- If poppler is missing, install it with:

```bash
conda install -c conda-forge poppler -y
```

- If some PDFs fail, the script continues with remaining files and prints failed filenames.
