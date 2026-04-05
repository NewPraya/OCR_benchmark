#!/bin/bash

# Exit on the first error.
set -e

# Resolve script directory and run from it.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=================================================="
echo "PDF to PNG Conversion Setup and Execution Script"
echo "=================================================="
echo "Working in: $SCRIPT_DIR"

# Activate conda environment.
echo -e "\n[1/4] Activating conda environment: OCR_benchmark"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate OCR_benchmark

# Install poppler (system dependency for pdf2image).
echo -e "\n[2/4] Installing poppler (system dependency)..."
conda install -c conda-forge poppler -y

# Install Python dependencies.
echo -e "\n[3/4] Installing Python dependencies..."
pip install -r requirements_pdf_convert.txt

# Run conversion.
echo -e "\n[4/4] Running PDF to PNG conversion..."
python convert_pdf_to_png.py

echo -e "\n=================================================="
echo "✓ All done! Check data/prepare/png_output/ for the converted PNG files."
echo "=================================================="
