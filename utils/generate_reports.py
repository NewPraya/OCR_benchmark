#!/usr/bin/env python3
"""
Batch generate report_*.json from existing preds_*.json files.
Usage: python utils/generate_reports.py [--version v1|v2|all]
"""
import json
import os
import sys
import glob
import argparse

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from evaluators.evaluator import OCREvaluator
from evaluators.evaluator_v2 import OCREvaluatorV2

def get_gt_path(version):
    if version == "v2":
        return "data/sample_gt_v2.json"
    return "data/sample_gt_v1.json" if os.path.exists("data/sample_gt_v1.json") else "data/sample_gt.json"

def generate_reports_for_version(version, no_postprocess=False):
    mode = "no-postprocess ablation" if no_postprocess else "default"
    print(f"\n🔄 Generating reports for {version.upper()} ({mode})...")
    
    gt_path = get_gt_path(version)
    if not os.path.exists(gt_path):
        print(f"  ⚠️  Ground truth not found: {gt_path}")
        return 0
    
    # Initialize evaluator
    evaluator = (
        OCREvaluatorV2(gt_path, enable_postprocess=(not no_postprocess))
        if version == "v2"
        else OCREvaluator(gt_path, normalize=(not no_postprocess))
    )
    
    # Find all prediction files
    pred_files = sorted(glob.glob(f"results/preds_{version}_*.json"))
    if not pred_files:
        print(f"  ℹ️  No prediction files found for {version.upper()}")
        return 0
    
    count = 0
    for pred_file in pred_files:
        model_id = os.path.basename(pred_file).replace(f"preds_{version}_", "").replace(".json", "")
        if no_postprocess:
            if model_id.endswith("__no_post"):
                print(f"  ⏭️  Skip already-tagged no_post file: {pred_file}")
                continue
            output_model_id = f"{model_id}__no_post"
        else:
            output_model_id = model_id
        report_path = f"results/report_{version}_{model_id}.json"
        if no_postprocess:
            report_path = f"results/report_{version}_{output_model_id}.json"
        
        # Skip if report already exists
        if os.path.exists(report_path):
            print(f"  ⏭️  Report exists: {report_path}")
            continue
        
        # Load predictions
        try:
            with open(pred_file, 'r') as f:
                predictions = json.load(f)
            
            if not predictions:
                print(f"  ⚠️  Empty predictions: {pred_file}")
                continue
            
            # Evaluate
            report = evaluator.evaluate_results(predictions)
            report['model_id'] = output_model_id
            report['postprocess_enabled'] = (not no_postprocess)
            
            # Save report
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)

            # Duplicate predictions file for dashboard detailed-view compatibility.
            if no_postprocess:
                tagged_pred_path = f"results/preds_{version}_{output_model_id}.json"
                if not os.path.exists(tagged_pred_path):
                    with open(tagged_pred_path, 'w') as pf:
                        json.dump(predictions, pf, indent=2)
            
            print(f"  ✅ Generated: {report_path}")
            count += 1
            
        except Exception as e:
            print(f"  ❌ Error processing {pred_file}: {e}")
    
    return count

def main():
    parser = argparse.ArgumentParser(description="Generate report files from predictions")
    parser.add_argument(
        "--version",
        type=str,
        default="all",
        choices=["v1", "v2", "all"],
        help="Which version to generate reports for (default: all)"
    )
    parser.add_argument(
        "--no-postprocess",
        action="store_true",
        help="Generate ablation reports without evaluator post-processing"
    )
    args = parser.parse_args()
    
    # Change to project root directory
    os.chdir(project_root)
    os.makedirs("results", exist_ok=True)
    
    total = 0
    if args.version in ("v1", "all"):
        total += generate_reports_for_version("v1", no_postprocess=args.no_postprocess)
    if args.version in ("v2", "all"):
        total += generate_reports_for_version("v2", no_postprocess=args.no_postprocess)
    
    print(f"\n✨ Done! Generated {total} report(s).")

if __name__ == "__main__":
    main()
