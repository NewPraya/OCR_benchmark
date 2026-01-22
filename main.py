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
from evaluators.schema_evaluator import SchemaBasedEvaluator
from schemas.schema_base import SchemaLoader
from utils.prompts import DEFAULT_PROMPTS

def run_benchmark(model_type, model_ids, eval_version="v1", gt_path=None, schema_path=None, image_dir="data/"):
    # Determine default GT path if not provided
    if gt_path is None:
        gt_path = "data/sample_gt_v2.json" if eval_version == "v2" else "data/sample_gt.json"
    
    # Get prompt and schema based on version/path
    schema = None
    if eval_version == "v2" and schema_path:
        print(f"📂 Using Schema: {schema_path}")
        schema = SchemaLoader.load_schema(schema_path)
        prompt = schema.get_prompt()
    else:
        # Fallback to legacy hardcoded prompts
        prompt = DEFAULT_PROMPTS.get(eval_version)
    
    # Load Ground Truth
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)

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
        
        # Run Predictions
        predictions = []
        for item in gt_data:
            file_name = item['file_name']
            image_path = os.path.join(image_dir, file_name)
            
            print(f"  - Processing {file_name}...")
            try:
                pred_text = model.predict(image_path, prompt)
                predictions.append({
                    "file_name": file_name,
                    "prediction": pred_text,
                    "model_name": model.model_name
                })
            except Exception as e:
                print(f"  ❌ Error processing {file_name} with {mid}: {e}")
        
        # Save Predictions
        os.makedirs("results", exist_ok=True)
        # Add version to filename to avoid overwriting
        output_path = f"results/preds_{eval_version}_{mid.replace('/', '_')}.json"
        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        # Evaluate
        if eval_version == "v2":
            if schema:
                # Use Schema-based Generic Evaluator
                evaluator = SchemaBasedEvaluator(gt_path, schema)
                report = evaluator.evaluate_results(predictions)
                print_report_schema(model, report, output_path)
            else:
                # Use Legacy Medical Form Evaluator
                evaluator = OCREvaluatorV2(gt_path)
                report = evaluator.evaluate_results(predictions)
                print_report_v2(model, report, output_path)
        else:
            evaluator = OCREvaluator(gt_path)
            report = evaluator.evaluate_results(predictions)
            print_report_v1(model, report, output_path)
            
        report['model_id'] = mid
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
    print(f"Avg Logical Acc: {report['avg_logical_acc']:.4f}")
    print(f"Avg Disease Acc: {report['avg_disease_acc']:.4f}")
    print(f"Avg Entity Recall: {report['avg_entity_recall']:.4f}")
    print(f"Avg Pairing Acc: {report['avg_pairing_acc']:.4f}")
    print("-" * 70)
    print(f"{'File Name':<20} | {'LogAcc':<7} | {'DisAcc':<7} | {'Recall':<7} | {'PairAcc':<7}")
    print("-" * 70)
    for detail in report['details']:
        print(f"{detail['file_name']:<20} | {detail['logical_acc']:<7.2f} | {detail['disease_acc']:<7.2f} | {detail['entity_recall']:<7.2f} | {detail['pairing_acc']:<7.2f}")
    print("="*70)
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
    parser.add_argument("-s", "--schema", type=str, default=None, help="Path to schema YAML (for V2 mode)")
    parser.add_argument("--gt", type=str, default=None, help="Custom GT JSON path")
    args = parser.parse_args()

    run_benchmark(args.model, args.model_id, eval_version=args.version, gt_path=args.gt, schema_path=args.schema)

if __name__ == "__main__":
    main()
