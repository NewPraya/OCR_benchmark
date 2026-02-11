import json
import os
import argparse
from models.dummy_model import DummyOCRModel
from models.gemini_model import GeminiOCRModel
from models.qwen_model import QwenOCRModel
from models.openai_model import OpenAIOCRModel
from models.ollama_model import OllamaOCRModel
from evaluators.evaluator import OCREvaluator
from evaluators.evaluator_v2 import OCREvaluatorV2
from utils.prompts import DEFAULT_PROMPTS
from utils.dataset_splits import load_splits, get_split_for_version, filter_gt_data

def _default_gt_path(eval_version: str) -> str:
    if eval_version == "v2":
        return "data/sample_gt_v2.json"
    # Prefer v1-specific file if present
    return "data/sample_gt_v1.json" if os.path.exists("data/sample_gt_v1.json") else "data/sample_gt.json"

def _load_existing_predictions(output_path: str):
    if not os.path.exists(output_path):
        return []
    try:
        with open(output_path, 'r') as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []

def _sort_predictions_by_gt_order(predictions, gt_data):
    """Sort predictions to match GT file order, then append unknown files."""
    gt_order = {item.get("file_name"): idx for idx, item in enumerate(gt_data)}
    return sorted(
        predictions,
        key=lambda p: (
            gt_order.get(p.get("file_name"), float("inf")),
            str(p.get("file_name", "")),
        ),
    )

def _report_output_path(eval_version: str, model_id: str) -> str:
    safe_model_id = model_id.replace("/", "_")
    return f"results/report_{eval_version}_{safe_model_id}.json"

def run_benchmark(model_type, model_ids, eval_version="v1", gt_path=None, schema_path=None, image_dir="data/", resume=True, split_path=None):
    # Determine default GT path if not provided
    if gt_path is None:
        gt_path = _default_gt_path(eval_version)
    
    # Get prompt based on version (v2 is now format-agnostic)
    if eval_version == "v2" and schema_path:
        print("⚠️  Schema-based v2 is deprecated. Ignoring schema and using simple prompt.")
    prompt = DEFAULT_PROMPTS.get(eval_version)
    
    # Load Ground Truth
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
    splits = load_splits(split_path)
    split_set = get_split_for_version(splits, eval_version)
    if split_set:
        gt_data = filter_gt_data(gt_data, split_set)
        print(f"🔎 Using split list: {len(gt_data)} images for {eval_version.upper()}")

    all_reports = []

    # Iterate over each model ID
    for mid in model_ids:
        print(f"\n🚀 Running Benchmark ({eval_version.upper()}) for Model: {mid}")
        
        # Initialize Model
        if model_type == "dummy":
            model = DummyOCRModel()
        elif model_type == "gemini":
            model = GeminiOCRModel(model_id=mid)
        elif model_type == "qwen":
            model = QwenOCRModel(model_id=mid)
        elif model_type == "openai":
            model = OpenAIOCRModel(model_id=mid)
        elif model_type == "ollama":
            model = OllamaOCRModel(model_id=mid)
        else:
            print(f"Unknown model type: {model_type}")
            continue
        
        # Run Predictions (resume-capable)
        os.makedirs("results", exist_ok=True)
        output_path = f"results/preds_{eval_version}_{mid.replace('/', '_')}.json"
        predictions = _load_existing_predictions(output_path) if resume else []
        seen_files = {p.get("file_name") for p in predictions if isinstance(p, dict)}
        if resume and seen_files:
            print(f"  ⏩ Resuming: {len(seen_files)} already processed.")

        for item in gt_data:
            file_name = item['file_name']
            image_path = os.path.join(image_dir, file_name)
            if resume and file_name in seen_files:
                continue
            
            print(f"  - Processing {file_name}...")
            try:
                pred_text = model.predict(image_path, prompt)
                predictions.append({
                    "file_name": file_name,
                    "prediction": pred_text,
                    "model_name": model.model_name
                })
                with open(output_path, 'w') as f:
                    json.dump(predictions, f, indent=2)
            except Exception as e:
                print(f"  ❌ Error processing {file_name} with {mid}: {e}")
        
        # Keep result file order stable and aligned with GT order.
        predictions = _sort_predictions_by_gt_order(predictions, gt_data)

        # Save Predictions (final write)
        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        # Evaluate
        if eval_version == "v2":
            evaluator = OCREvaluatorV2(gt_path)
            report = evaluator.evaluate_results(predictions)
            print_report_v2(model, report, output_path)
        else:
            evaluator = OCREvaluator(gt_path)
            report = evaluator.evaluate_results(predictions)
            print_report_v1(model, report, output_path)
            
        report['model_id'] = mid
        report_path = _report_output_path(eval_version, mid)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Summary report saved to: {report_path}")
        all_reports.append(report)
    
    return all_reports

def print_report_v1(model, report, output_path):
    print("\n" + "="*50)
    print(f"V1 REPORT: {model.model_name}")
    print("="*50)
    print(f"Samples: {report['sample_count']}")
    print(f"Avg CER: {report['average_cer']:.4f}")
    print(f"Avg WER: {report['average_wer']:.4f}")
    print("-" * 50)
    print(f"{'File Name':<20} | {'CER':<8} | {'WER':<8}")
    print("-" * 50)
    for detail in report['details']:
        print(f"{detail['file_name']:<20} | {detail['cer']:<8.4f} | {detail['wer']:<8.4f}")
    print("="*50)
    print(f"Full results saved to: {output_path}")

def print_report_v2(model, report, output_path):
    print("\n" + "="*70)
    print(f"V2 REPORT: {model.model_name}")
    print("="*70)
    print(f"Samples: {report['sample_count']}")
    print(f"Avg Y/N Acc: {report['avg_yn_acc']:.4f}")
    print(f"Avg Handwriting CER: {report['avg_handwriting_cer']:.4f}")
    print(f"Avg Handwriting WER: {report['avg_handwriting_wer']:.4f}")
    print(f"Avg Handwriting NED: {report['avg_handwriting_ned']:.4f}")
    print(f"Avg Weighted Score: {report['avg_weighted_score']:.4f}")
    print("-" * 70)
    print(f"{'File Name':<20} | {'YNAcc':<7} | {'CER':<7} | {'WER':<7} | {'NED':<7}")
    print("-" * 70)
    for detail in report['details']:
        print(f"{detail['file_name']:<20} | {detail['yn_acc']:<7.2f} | {detail['handwriting_cer']:<7.2f} | {detail['handwriting_wer']:<7.2f} | {detail['handwriting_ned']:<7.2f}")
    print("="*70)
    if report.get("yn_question_stats"):
        print("Top Y/N error questions (lowest accuracy):")
        for item in report["yn_question_stats"][:10]:
            print(f"  - {item['label']}: {item['correct']}/{item['total']} (acc={item['accuracy']:.2f})")
    print(f"Full results saved to: {output_path}")

def print_report_schema(model, report, output_path):
    print("\n" + "="*70)
    print(f"V2 SCHEMA REPORT: {model.model_name} (Schema: {report.get('schema_name')})")
    print("="*70)
    print(f"Samples: {report['sample_count']}")
    print(f"Overall Weighted Score: {report['avg_weighted_score']:.4f}")
    print("-" * 70)
    # Dynamically print field scores from schema
    for field_name, score in report.items():
        if field_name.startswith('avg_') and field_name != 'avg_weighted_score' and not isinstance(score, (list, dict)):
            print(f"{field_name[4:]:>20}: {score:.4f}")
    print("="*70)
    print(f"Full results saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="OCR Benchmark Runner")
    parser.add_argument("-m", "--model", type=str, default="dummy", choices=["dummy", "gemini", "qwen", "openai", "ollama"], help="Model type")
    parser.add_argument("-id", "--model_id", type=str, nargs="+", default=["gemini-2.0-flash-exp"], help="One or more Model IDs")
    parser.add_argument("-v", "--version", type=str, default="v1", choices=["v1", "v2"], help="Evaluation version (v1=text, v2=structured)")
    parser.add_argument("-s", "--schema", type=str, default=None, help="Path to schema YAML (deprecated in V2)")
    parser.add_argument("--gt", type=str, default=None, help="Custom GT JSON path")
    parser.add_argument("--split", type=str, default=None, help="Optional split JSON (v1/v2 file lists)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing predictions file")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Disable resume")
    parser.set_defaults(resume=True)
    args = parser.parse_args()

    run_benchmark(
        args.model,
        args.model_id,
        eval_version=args.version,
        gt_path=args.gt,
        schema_path=args.schema,
        resume=args.resume,
        split_path=args.split
    )

if __name__ == "__main__":
    main()
