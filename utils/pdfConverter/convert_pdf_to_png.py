#!/usr/bin/env python3
"""
Convert all PDFs in data/prepare/ to PNG format and rename them sequentially starting from 6.
"""

import os
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image

def convert_pdfs_to_pngs(input_dir, output_dir, start_number=6):
    """
    Convert all PDFs in input_dir to PNG and save with sequential numbering.
    
    Args:
        input_dir: Directory containing PDF files
        output_dir: Directory to save PNG files
        start_number: Starting number for sequential naming (default: 6)
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all PDF files
    pdf_files = sorted([f for f in Path(input_dir).glob('*.pdf')])
    
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF files")
    print(f"Converting PDFs to PNG and renaming starting from {start_number}...")
    
    current_number = start_number
    
    for pdf_file in pdf_files:
        try:
            print(f"\nProcessing: {pdf_file.name}")
            
            # Convert PDF to images (using first page only)
            images = convert_from_path(
                pdf_file,
                dpi=300,  # High quality
                first_page=1,
                last_page=1
            )
            
            if images:
                # Save as PNG with sequential number
                output_filename = f"{current_number}.png"
                output_filepath = output_path / output_filename
                
                # Save the first page
                images[0].save(output_filepath, 'PNG')
                print(f"  ✓ Saved as: {output_filename}")
                
                current_number += 1
            else:
                print(f"  ✗ No images extracted from {pdf_file.name}")
                
        except Exception as e:
            print(f"  ✗ Error processing {pdf_file.name}: {str(e)}")
            continue
    
    print(f"\n{'='*50}")
    print(f"Conversion complete!")
    print(f"Total files processed: {current_number - start_number}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*50}")

if __name__ == "__main__":
    # Get the directory where the script is located
    SCRIPT_DIR = Path(__file__).parent.absolute()
    # Assuming project root is two levels up from utils/pdfConverter/
    PROJECT_ROOT = SCRIPT_DIR.parent.parent
    
    input_directory = PROJECT_ROOT / "data/prepare"
    output_directory = input_directory / "png_output"
    
    convert_pdfs_to_pngs(input_directory, output_directory, start_number=6)
