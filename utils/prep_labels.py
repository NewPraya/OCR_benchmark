import os
import json
import glob
import sys
import argparse

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.gemini_model import GeminiOCRModel
from utils.prompts import DEFAULT_PROMPTS

def prep_labels(version="v1"):
    """
    Calls Gemini to draft OCR results and saves them as Markdown files for human editing.
    """
    prompt = DEFAULT_PROMPTS.get(version)
    
    try:
        model = GeminiOCRModel(model_id='gemini-3-flash-preview')
    except Exception as e:
        print(f"❌ Failed to initialize Gemini model: {e}")
        return
    
    image_dir = "data"
    label_dir = f"labeling_{version}"
    os.makedirs(label_dir, exist_ok=True)

    extensions = ['*.png', '*.jpg', '*.jpeg', '*.webp']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))

    for img_path in image_files:
        file_name = os.path.basename(img_path)
        label_file = os.path.join(label_dir, f"{file_name}.md")

        if os.path.exists(label_file):
            print(f"⏩ Skipping {file_name}, label file already exists.")
            continue

        print(f"🤖 Gemini is drafting ({version.upper()}) for {file_name}...")
        try:
            prediction = model.predict(img_path, prompt)
            
            # Clean up prediction if it contains markdown code blocks
            clean_pred = prediction
            if "```json" in clean_pred:
                clean_pred = clean_pred.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_pred:
                clean_pred = clean_pred.split("```")[1].split("```")[0].strip()
            
            rel_img_path = os.path.relpath(img_path, label_dir)
            
            if version == "v2":
                content = f"""# Labeling V2: {file_name}
![{file_name}]({rel_img_path})

---
### Ground Truth (Edit below)
```json
{clean_pred}
```
"""
            else:
                content = f"""# Labeling V1: {file_name}
![{file_name}]({rel_img_path})

---
### Ground Truth (Edit below)
{prediction}
"""
            with open(label_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Draft created: {label_file}")
        except Exception as e:
            print(f"❌ Error drafting {file_name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version", type=str, default="v1", choices=["v1", "v2"])
    args = parser.parse_args()
    prep_labels(version=args.version)
