import os
import json
import glob
import sys
import argparse

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

def sync_to_gt(version="v1"):
    """
    Reads edited Markdown files and syncs them back to the appropriate GT JSON.
    """
    label_dir = f"labeling_{version}"
    gt_path = "data/sample_gt_v2.json" if version == "v2" else "data/sample_gt.json"
    
    if not os.path.exists(label_dir):
        print(f"Error: Labeling directory '{label_dir}' not found.")
        return

    results = []
    md_files = glob.glob(os.path.join(label_dir, "*.md"))
    
    for md_path in md_files:
        file_name = os.path.basename(md_path).rsplit('.md', 1)[0]
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
            header = "### Ground Truth (Edit below)"
            if header in content:
                raw_text = content.split(header)[1].strip()
                
                if version == "v2":
                    # Robust JSON extraction
                    json_str = raw_text
                    
                    # If multiple code blocks exist, find the one that looks like JSON
                    if "{" in json_str and "}" in json_str:
                        # Extract everything between the first '{' and the last '}'
                        start = json_str.find("{")
                        end = json_str.rfind("}") + 1
                        json_str = json_str[start:end]
                    
                    try:
                        data = json.loads(json_str)
                        data['file_name'] = file_name
                        results.append(data)
                    except Exception as e:
                        print(f"⚠️ JSON Error in {md_path}: {e}")
                        print(f"--- Raw Content (trimmed) ---\n{json_str[:100]}...")
                else:
                    # V1 text format
                    results.append({"file_name": file_name, "text": raw_text})

    results.sort(key=lambda x: x.get('file_name', ''))
    with open(gt_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"🚀 Synced {len(results)} {version.upper()} labels to {gt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version", type=str, default="v1", choices=["v1", "v2"])
    args = parser.parse_args()
    sync_to_gt(version=args.version)
