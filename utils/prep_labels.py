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
from utils.dataset_splits import load_splits, get_split_for_version, filter_file_names

def _load_gt_file_names(gt_path: str):
    if not gt_path:
        return None
    if not os.path.exists(gt_path):
        print(f"⚠️  GT file not found: {gt_path}. Falling back to image folder scan.")
        return None
    try:
        with open(gt_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        return [item.get("file_name") for item in gt_data if item.get("file_name")]
    except Exception as e:
        print(f"⚠️  Failed to read GT file: {e}. Falling back to image folder scan.")
        return None

def _load_progress(progress_path: str):
    if not os.path.exists(progress_path):
        return {"completed": [], "failed": {}}
    try:
        with open(progress_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"completed": [], "failed": {}}
        data.setdefault("completed", [])
        data.setdefault("failed", {})
        return data
    except Exception:
        return {"completed": [], "failed": {}}

def _save_progress(progress_path: str, progress: dict):
    tmp_path = progress_path + ".tmp"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, progress_path)

def prep_labels(version="v1", gt_path=None, image_dir="data", split_path=None):
    """
    Calls Gemini to draft OCR results and saves them as Markdown files for human editing.
    """
    prompt = DEFAULT_PROMPTS.get(version)
    
    try:
        model = GeminiOCRModel(model_id='gemini-3-flash-preview')
    except Exception as e:
        print(f"❌ Failed to initialize Gemini model: {e}")
        return
    
    label_dir = f"labeling_{version}"
    os.makedirs(label_dir, exist_ok=True)
    progress_path = os.path.join(label_dir, "_progress.json")
    progress = _load_progress(progress_path)

    splits = load_splits(split_path)
    split_set = get_split_for_version(splits, version)
    gt_file_names = _load_gt_file_names(gt_path)
    if gt_file_names:
        gt_file_names = filter_file_names(gt_file_names, split_set)
        image_files = [os.path.join(image_dir, fn) for fn in gt_file_names]
    else:
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.webp']
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        if split_set:
            image_files = [p for p in image_files if os.path.basename(p) in split_set]

    if split_set:
        print(f"🔎 Using split list: {len(image_files)} images for {version.upper()}")

    for img_path in image_files:
        file_name = os.path.basename(img_path)
        label_file = os.path.join(label_dir, f"{file_name}.md")

        if os.path.exists(label_file):
            print(f"⏩ Skipping {file_name}, label file already exists.")
            if file_name not in progress["completed"]:
                progress["completed"].append(file_name)
                _save_progress(progress_path, progress)
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
            if file_name not in progress["completed"]:
                progress["completed"].append(file_name)
            if file_name in progress["failed"]:
                progress["failed"].pop(file_name, None)
            _save_progress(progress_path, progress)
        except Exception as e:
            print(f"❌ Error drafting {file_name}: {e}")
            progress["failed"][file_name] = str(e)
            _save_progress(progress_path, progress)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version", type=str, default="v1", choices=["v1", "v2"])
    parser.add_argument("--gt", type=str, default=None, help="Optional GT JSON path to limit images")
    parser.add_argument("--image_dir", type=str, default="data", help="Image folder (default: data)")
    parser.add_argument("--split", type=str, default=None, help="Optional split JSON (v1/v2 file lists)")
    args = parser.parse_args()
    prep_labels(version=args.version, gt_path=args.gt, image_dir=args.image_dir, split_path=args.split)
