import json
import os
import argparse
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
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

def _pred_output_path(eval_version: str, model_id: str) -> str:
    safe_model_id = model_id.replace("/", "_")
    return f"results/preds_{eval_version}_{safe_model_id}.json"

def _multirun_pred_output_path(eval_version: str, model_id: str, run_index: int) -> str:
    safe_model_id = model_id.replace("/", "_")
    return f"results/multirun/preds_{eval_version}_{safe_model_id}__run{run_index}.json"

def _multirun_meta_output_path(eval_version: str, model_id: str) -> str:
    safe_model_id = model_id.replace("/", "_")
    return f"results/multirun/meta_{eval_version}_{safe_model_id}.json"

def _variant_model_id(model_id: str, postprocess: bool) -> str:
    return model_id if postprocess else f"{model_id}__no_post"

def _package_version(pkg_name: str):
    try:
        return importlib_metadata.version(pkg_name)
    except Exception:
        return None

def _collect_runtime_metadata(model_type: str, model_id: str):
    return {
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
        "model_type": model_type,
        "model_id": model_id,
        "sdk_versions": {
            "openai": _package_version("openai"),
            "google-genai": _package_version("google-genai"),
            "python-dotenv": _package_version("python-dotenv"),
        },
        "api_base_urls": {
            "openai": os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1",
            "qwen_dashscope_compatible": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        },
    }

def _is_transient_network_error(exc: Exception) -> bool:
    """Best-effort detection for transient network/provider errors."""
    msg = str(exc).lower()
    network_signals = (
        "timeout",
        "timed out",
        "connection",
        "dns",
        "name or service not known",
        "nodename nor servname",
        "temporary failure",
        "network is unreachable",
        "connection reset",
        "connection aborted",
        "connection refused",
        "server disconnected without sending a response",
        "server disconnected",
        "remoteprotocolerror",
        "peer closed connection",
        "ssl",
        "tls",
        "rate limit",
        "too many requests",
        "status=429",
        "status=500",
        "status=502",
        "status=503",
        "status=504",
    )
    if any(s in msg for s in network_signals):
        return True

    # Common built-in transient categories.
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    return False

def _collect_processing_stats(predictions, gt_data):
    gt_files = {item.get("file_name") for item in gt_data if isinstance(item, dict) and item.get("file_name")}
    pred_entries = [p for p in predictions if isinstance(p, dict)]
    processed_files = {p.get("file_name") for p in pred_entries if p.get("file_name") in gt_files}
    failed_files = {
        p.get("file_name")
        for p in pred_entries
        if p.get("file_name") in gt_files and bool(p.get("failed"))
    }
    missing_files = gt_files - processed_files
    failed_files |= missing_files
    success_files = processed_files - failed_files
    target_count = len(gt_files)
    failed_count = len(failed_files)
    return {
        "target_count": target_count,
        "processed_count": len(processed_files),
        "success_count": len(success_files),
        "failed_count": failed_count,
        "failed_rate": (failed_count / target_count) if target_count else 0.0,
        "failed_files": sorted(failed_files),
    }

def run_benchmark(
    model_type,
    model_ids,
    eval_version="v1",
    gt_path=None,
    schema_path=None,
    image_dir="data/",
    resume=True,
    split_path=None,
    postprocess=True,
    runs_per_image=1
):
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
        variant_mid = _variant_model_id(mid, postprocess)
        safe_variant_mid = variant_mid.replace("/", "_")
        print(f"\n🚀 Running Benchmark ({eval_version.upper()}) for Model: {mid}")
        if not postprocess:
            print("  🧪 Ablation mode: post-processing disabled")
        if runs_per_image > 1:
            print("  🔁 Multi-run mode enabled: "
                  f"{runs_per_image} runs/image (run1 will be used for dashboard-compatible outputs)")
        
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
        
        if runs_per_image < 1:
            raise ValueError("runs_per_image must be >= 1")

        # Run Predictions (resume-capable, optionally multi-run)
        os.makedirs("results", exist_ok=True)
        output_path = _pred_output_path(eval_version, safe_variant_mid)
        run_predictions_map = {}
        run_output_paths = {}

        if runs_per_image > 1:
            os.makedirs("results/multirun", exist_ok=True)

        for run_index in range(1, runs_per_image + 1):
            run_output_path = (
                _multirun_pred_output_path(eval_version, safe_variant_mid, run_index)
                if runs_per_image > 1
                else output_path
            )
            predictions = _load_existing_predictions(run_output_path) if resume else []
            # Bootstrap primary run from legacy single-run file for backwards-compatible resume.
            if (
                runs_per_image > 1
                and resume
                and run_index == 1
                and not predictions
                and os.path.exists(output_path)
            ):
                predictions = _load_existing_predictions(output_path)
                if predictions:
                    print(f"  ♻️ Bootstrapped run {run_index} from legacy file: {output_path}")

            seen_files = {p.get("file_name") for p in predictions if isinstance(p, dict)}
            if resume and seen_files:
                run_label = f"run {run_index}" if runs_per_image > 1 else "single run"
                print(f"  ⏩ Resuming {run_label}: {len(seen_files)} already processed.")

            if runs_per_image > 1:
                print(f"  ▶️ Starting run {run_index}/{runs_per_image}...")

            for item in gt_data:
                file_name = item['file_name']
                image_path = os.path.join(image_dir, file_name)
                if resume and file_name in seen_files:
                    continue
                
                if runs_per_image > 1:
                    print(f"    - [run {run_index}] Processing {file_name}...")
                else:
                    print(f"  - Processing {file_name}...")
                try:
                    pred_text = model.predict(image_path, prompt)
                    predictions.append({
                        "file_name": file_name,
                        "prediction": pred_text,
                        "model_name": model.model_name,
                        "failed": False,
                    })
                    with open(run_output_path, 'w') as f:
                        json.dump(predictions, f, indent=2)
                except Exception as e:
                    if _is_transient_network_error(e):
                        print(f"  🌐 Transient network error on {file_name} with {mid}: {e}")
                        print("    ↪️ Skip recording this sample; it will be retried on next --resume run.")
                        continue

                    print(f"  ❌ Error processing {file_name} with {mid}: {e}")
                    predictions.append({
                        "file_name": file_name,
                        "prediction": "",
                        "model_name": model.model_name,
                        "failed": True,
                        "error": str(e),
                    })
                    with open(run_output_path, 'w') as f:
                        json.dump(predictions, f, indent=2)
            
            predictions = _sort_predictions_by_gt_order(predictions, gt_data)
            with open(run_output_path, 'w') as f:
                json.dump(predictions, f, indent=2)
            run_predictions_map[run_index] = predictions
            run_output_paths[run_index] = run_output_path

        # Keep dashboard compatibility by exposing one primary run via legacy path.
        predictions = run_predictions_map[1]
        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=2)

        runtime_metadata = _collect_runtime_metadata(model_type=model_type, model_id=variant_mid)
        if runs_per_image > 1:
            meta_path = _multirun_meta_output_path(eval_version, safe_variant_mid)
            meta_payload = {
                "eval_version": eval_version,
                "model_id": variant_mid,
                "postprocess_enabled": postprocess,
                "runs_per_image": runs_per_image,
                "primary_pred_path": output_path,
                "run_pred_paths": {str(k): v for k, v in run_output_paths.items()},
                "runtime": runtime_metadata,
            }
            with open(meta_path, "w") as f:
                json.dump(meta_payload, f, indent=2)
            print(f"Multi-run metadata saved to: {meta_path}")
        
        # Evaluate
        if eval_version == "v2":
            evaluator = OCREvaluatorV2(gt_path, enable_postprocess=postprocess)
            report = evaluator.evaluate_results(predictions)
            print_report_v2(model, report, output_path)
        else:
            evaluator = OCREvaluator(gt_path, normalize=postprocess)
            report = evaluator.evaluate_results(predictions)
            print_report_v1(model, report, output_path)

        report.update(_collect_processing_stats(predictions, gt_data))
        report['model_id'] = variant_mid
        report["postprocess_enabled"] = postprocess
        report["runs_per_image"] = runs_per_image
        report["runtime"] = runtime_metadata
        report_path = _report_output_path(eval_version, variant_mid)
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
    print(f"Failed: {report.get('failed_count', 0)} / {report.get('target_count', report['sample_count'])} ({report.get('failed_rate', 0.0):.2%})")
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
    print(f"Failed: {report.get('failed_count', 0)} / {report.get('target_count', report['sample_count'])} ({report.get('failed_rate', 0.0):.2%})")
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
    parser.add_argument("--no-postprocess", action="store_true", help="Disable evaluator post-processing (ablation)")
    parser.add_argument("--runs-per-image", type=int, default=1, help="How many independent runs per image (default: 1)")
    parser.set_defaults(resume=True)
    args = parser.parse_args()

    run_benchmark(
        args.model,
        args.model_id,
        eval_version=args.version,
        gt_path=args.gt,
        schema_path=args.schema,
        resume=args.resume,
        split_path=args.split,
        postprocess=(not args.no_postprocess),
        runs_per_image=args.runs_per_image
    )

if __name__ == "__main__":
    main()
