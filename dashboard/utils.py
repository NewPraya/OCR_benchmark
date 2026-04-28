import glob
import io
import json
import os
import zipfile

import pandas as pd

from evaluators.statistical_tests import batch_compare_models


def resolve_gt_path(v_key: str) -> str:
    if v_key == "v2":
        return "data/sample_gt_v2.json"
    return "data/sample_gt_v1.json" if os.path.exists("data/sample_gt_v1.json") else "data/sample_gt.json"


def has_optional_reproduction_assets() -> bool:
    required = [
        "results/multirun/leaderboard_v1.json",
        "results/multirun/leaderboard_v2.json",
        "results/multirun/distribution_v1.json",
        "results/multirun/distribution_v2.json",
    ]
    return all(os.path.exists(path) for path in required)


def file_signature(path: str):
    if not os.path.exists(path):
        return (path, 0.0, 0)
    stat = os.stat(path)
    return (path, stat.st_mtime, stat.st_size)


def result_file_signatures(v_key: str):
    result_files = sorted(glob.glob(f"results/preds_{v_key}_*.json"))
    return tuple(file_signature(path) for path in result_files)


def report_file_signatures(v_key: str):
    report_files = sorted(glob.glob(f"results/report_{v_key}_*.json"))
    return tuple(file_signature(path) for path in report_files)


def save_report_file(v_key: str, model_id: str, report: dict):
    os.makedirs("results", exist_ok=True)
    safe_model_id = model_id.replace("/", "_")
    report_path = f"results/report_{v_key}_{safe_model_id}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)


def resolve_pred_path(v_key: str, model_id: str) -> str:
    candidates = [
        f"results/preds_{v_key}_{model_id}.json",
        f"results/preds_{v_key}_{model_id.replace('/', '_')}.json",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[1]


def base_model_id(model_id: str) -> str:
    return model_id[:-9] if model_id.endswith("__no_post") else model_id


def report_summary_row(v_key: str, model_id: str, report: dict, postprocess_default: bool = True) -> dict:
    common = {
        "Model ID": model_id,
        "Postprocess": report.get("postprocess_enabled", postprocess_default),
        "Samples": report.get("sample_count", 0),
        "Target": report.get("target_count", report.get("sample_count", 0)),
        "Processed": report.get("processed_count", report.get("sample_count", 0)),
        "Failed": report.get("failed_count", 0),
        "Failed Rate": round(report.get("failed_rate", 0.0), 4),
    }
    if v_key == "v2":
        return {
            **common,
            "Weighted Score": round(report.get("avg_weighted_score", 0.0), 4),
            "Y/N Acc": round(report.get("avg_yn_acc", 0.0), 4),
            "Y Pos Prec": round(report.get("yn_positive_precision", 0.0), 4),
            "Y Pos Recall": round(report.get("yn_positive_recall", 0.0), 4),
            "Y Pos F1": round(report.get("yn_positive_f1", 0.0), 4),
            "Y Bal Acc": round(report.get("yn_balanced_accuracy", 0.0), 4),
            "HW CER": round(report.get("avg_handwriting_cer", 0.0), 4),
            "HW WER": round(report.get("avg_handwriting_wer", 0.0), 4),
            "HW NED": round(report.get("avg_handwriting_ned", 0.0), 4),
        }
    return {
        **common,
        "Avg CER": round(report.get("average_cer", 0.0), 4),
        "Avg WER": round(report.get("average_wer", 0.0), 4),
        "Avg NED": round(report.get("average_ned", 0.0), 4),
        "Precision": round(report.get("average_precision", 0.0), 4),
        "Recall": round(report.get("average_recall", 0.0), 4),
        "BoW F1": round(report.get("average_bow_f1", 0.0), 4),
        "Exact Match": round(report.get("exact_match_accuracy", 0.0), 4),
    }


def model_family_from_id(model_id: str) -> str:
    base = base_model_id(model_id).lower()
    if base.startswith("gemini"):
        return "Gemini"
    if base.startswith("qwen"):
        return "Qwen"
    if base.startswith("gpt"):
        return "OpenAI"
    if base.startswith("claude"):
        return "Claude"
    if base.startswith("llama") or "ollama" in base:
        return "Ollama/Llama"
    return "Other"


def detail_quality_score(v_key: str, detail) -> float:
    """
    Convert per-sample detail metrics to a unified quality score (higher is better).
    v2 uses weighted_score directly; v1 uses (1 - CER).
    """
    if not isinstance(detail, dict):
        return 0.0
    if v_key == "v2":
        return float(detail.get("weighted_score", 0.0) or 0.0)
    cer = float(detail.get("cer", 1.0) or 1.0)
    return max(0.0, 1.0 - cer)


def build_case_finder_df(v_key: str, full_results: dict, model_a: str, model_b: str) -> pd.DataFrame:
    rep_a = full_results.get(model_a, {})
    rep_b = full_results.get(model_b, {})
    det_a = {d.get("file_name"): d for d in rep_a.get("details", []) if isinstance(d, dict)}
    det_b = {d.get("file_name"): d for d in rep_b.get("details", []) if isinstance(d, dict)}
    common = sorted(set(det_a.keys()) & set(det_b.keys()))
    rows = []

    for fname in common:
        a = det_a[fname]
        b = det_b[fname]
        qa = detail_quality_score(v_key, a)
        qb = detail_quality_score(v_key, b)
        row = {
            "file_name": fname,
            f"{model_a}_quality": qa,
            f"{model_b}_quality": qb,
            "avg_quality": (qa + qb) / 2.0,
            "abs_gap": abs(qa - qb),
            "winner": model_a if qa > qb else (model_b if qb > qa else "tie"),
        }
        if v_key == "v2":
            row.update(
                {
                    f"{model_a}_yn_acc": float(a.get("yn_acc", 0.0) or 0.0),
                    f"{model_b}_yn_acc": float(b.get("yn_acc", 0.0) or 0.0),
                    f"{model_a}_hw_cer": float(a.get("handwriting_cer", 0.0) or 0.0),
                    f"{model_b}_hw_cer": float(b.get("handwriting_cer", 0.0) or 0.0),
                }
            )
        else:
            row.update(
                {
                    f"{model_a}_cer": float(a.get("cer", 1.0) or 1.0),
                    f"{model_b}_cer": float(b.get("cer", 1.0) or 1.0),
                }
            )
        rows.append(row)

    return pd.DataFrame(rows)


def export_to_latex(df: pd.DataFrame, caption: str = "OCR Benchmark Results") -> str:
    return df.to_latex(index=False, float_format="%.4f", caption=caption, label="tab:results")


def build_export_bundle(v_key: str, df: pd.DataFrame, caption: str, full_results=None, stats_metric=None):
    """
    Build a zip bundle for paper writing.
    Includes leaderboard exports, raw reports, and optional pairwise stats.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{v_key}/leaderboard_{v_key}.csv", df.to_csv(index=False))
        zf.writestr(f"{v_key}/leaderboard_{v_key}.json", df.to_json(orient="records", indent=2))
        zf.writestr(f"{v_key}/leaderboard_{v_key}.tex", export_to_latex(df, caption=caption))

        for report_path in sorted(glob.glob(f"results/report_{v_key}_*.json")):
            try:
                with open(report_path, "rb") as f:
                    zf.writestr(f"{v_key}/reports/{os.path.basename(report_path)}", f.read())
            except Exception:
                continue

        if full_results and len(full_results) >= 2 and stats_metric:
            try:
                comparisons = batch_compare_models(full_results, stats_metric, use_parametric=True)
                rows = []
                for comp in comparisons:
                    rows.append(
                        {
                            "model1": comp.get("model1_id"),
                            "model2": comp.get("model2_id"),
                            "metric": comp.get("metric"),
                            "n_samples": comp.get("n_samples"),
                            "model1_mean": comp.get("model1", {}).get("mean"),
                            "model1_ci_low": comp.get("model1", {}).get("ci_95", (None, None))[0],
                            "model1_ci_high": comp.get("model1", {}).get("ci_95", (None, None))[1],
                            "model2_mean": comp.get("model2", {}).get("mean"),
                            "model2_ci_low": comp.get("model2", {}).get("ci_95", (None, None))[0],
                            "model2_ci_high": comp.get("model2", {}).get("ci_95", (None, None))[1],
                            "test": comp.get("statistical_test", {}).get("test"),
                            "p_value": comp.get("statistical_test", {}).get("p_value"),
                            "significant": comp.get("statistical_test", {}).get("significant"),
                            "winner": comp.get("winner"),
                        }
                    )
                if rows:
                    stats_df = pd.DataFrame(rows)
                    zf.writestr(f"{v_key}/stats/pairwise_stats_{stats_metric}.csv", stats_df.to_csv(index=False))
                    zf.writestr(f"{v_key}/stats/pairwise_stats_{stats_metric}.json", json.dumps(rows, indent=2))
            except Exception:
                pass

        manifest = {
            "version": v_key,
            "rows": int(len(df)),
            "columns": list(df.columns),
            "caption": caption,
        }
        zf.writestr(f"{v_key}/manifest.json", json.dumps(manifest, indent=2))

    buf.seek(0)
    return buf.getvalue()


def stats_metric_options(v_key: str):
    if v_key == "v2":
        return ["weighted_score", "yn_acc", "handwriting_cer", "handwriting_wer", "handwriting_ned"], "weighted_score"
    return ["cer", "wer", "ned", "bow_f1", "exact_match"], "cer"


def format_ci(ci_pair) -> str:
    low = ci_pair[0] if isinstance(ci_pair, (list, tuple)) and len(ci_pair) > 0 else None
    high = ci_pair[1] if isinstance(ci_pair, (list, tuple)) and len(ci_pair) > 1 else None
    if low is None or high is None:
        return "N/A"
    return f"[{low:.4f}, {high:.4f}]"


def build_pairwise_table(comparisons) -> pd.DataFrame:
    rows = []
    for comp in comparisons:
        test = comp.get("statistical_test", {})
        rows.append(
            {
                "Model 1": comp.get("model1_id", "N/A"),
                "Model 2": comp.get("model2_id", "N/A"),
                "M1 Mean": f"{comp.get('model1', {}).get('mean', 0.0):.4f}",
                "M1 95% CI": format_ci(comp.get("model1", {}).get("ci_95")),
                "M2 Mean": f"{comp.get('model2', {}).get('mean', 0.0):.4f}",
                "M2 95% CI": format_ci(comp.get("model2", {}).get("ci_95")),
                "p-value": f"{(test.get('p_value') if test.get('p_value') is not None else 0.0):.6f}",
                "Significant": "✓" if test.get("significant", False) else "✗",
                "Winner": comp.get("winner", "N/A"),
            }
        )
    return pd.DataFrame(rows)
