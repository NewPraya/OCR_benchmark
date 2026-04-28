#!/usr/bin/env python3
"""
Generate appendix LaTeX from multirun artifacts.

Output:
  /temp/multirun_appendix.tex
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dashboard.utils import base_model_id
from evaluators.evaluator import OCREvaluator
from evaluators.evaluator_v2 import OCREvaluatorV2


KNOWN_V1_CATEGORIES = ["hw", "pn", "cf", "hd", "rl", "ess", "numeric"]
CATEGORY_LABELS = {
    "hw": "Handwritten notes",
    "pn": "Progress notes",
    "cf": "Consent forms",
    "hd": "Health declarations",
    "rl": "Referral letters",
    "ess": "Epworth Scale",
    "numeric": "Medical history",
}


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def display_model_name(model_id: str) -> str:
    return base_model_id(model_id).replace("-preview", "")


def format_float(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def is_close(a: float, b: float, eps: float = 1e-12) -> bool:
    return abs(a - b) <= eps


def v1_category(file_name: str) -> str:
    stem = file_name.rsplit(".", 1)[0]
    if stem.isdigit():
        return "numeric"
    m = re.match(r"^([a-zA-Z]+)", stem)
    prefix = m.group(1).lower() if m else ""
    if prefix in KNOWN_V1_CATEGORIES:
        return prefix
    return "numeric"


def ordered_models_from_leaderboard(rows: List[dict], metric: str, ascending: bool) -> List[str]:
    on_rows = [r for r in rows if bool(r.get("Postprocess", True))]
    on_rows = [r for r in on_rows if not str(base_model_id(r.get("Model ID", ""))).startswith("dummy")]
    ordered = sorted(on_rows, key=lambda r: float(r.get(metric, 0.0)), reverse=not ascending)
    return [base_model_id(r["Model ID"]) for r in ordered]


def collect_full_rows(v_key: str) -> Tuple[List[dict], List[str], int]:
    rows = load_json(f"results/multirun/leaderboard_{v_key}.json")
    rows = [r for r in rows if isinstance(r, dict)]
    if v_key == "v1":
        rows = [r for r in rows if not str(base_model_id(r.get("Model ID", ""))).startswith("dummy")]

    metrics = (
        ["Avg CER", "Avg WER", "Avg NED", "Precision", "Recall", "BoW F1", "Exact Match"]
        if v_key == "v1"
        else ["Weighted Score", "Y/N Acc", "HW CER", "HW WER", "HW NED"]
    )
    order_metric = "Avg CER" if v_key == "v1" else "Weighted Score"
    ascending = v_key == "v1"
    model_order = ordered_models_from_leaderboard(rows, order_metric, ascending=ascending)

    row_map = {(base_model_id(r["Model ID"]), bool(r["Postprocess"])): r for r in rows}

    out = []
    for model_id in model_order:
        for pp in (True, False):
            row = row_map.get((model_id, pp))
            if row is None:
                continue
            out.append(
                {
                    "model": model_id,
                    "pp": pp,
                    "samples": int(row.get("Samples", 0) or 0),
                    "metrics": {m: float(row.get(m, 0.0) or 0.0) for m in metrics},
                }
            )

    n_samples = int(np.median([r["samples"] for r in out])) if out else 0
    return out, metrics, n_samples


def best_by_metric(rows: List[dict], metrics: List[str], lower_better: set) -> Dict[str, float]:
    best = {}
    for m in metrics:
        vals = [r["metrics"][m] for r in rows]
        if not vals:
            continue
        best[m] = min(vals) if m in lower_better else max(vals)
    return best


def render_full_table_transcription(rows: List[dict], metrics: List[str], n_samples: int) -> str:
    lower_better = {"Avg CER", "Avg WER", "Avg NED"}
    best_vals = best_by_metric(rows, metrics, lower_better)

    lines = []
    lines.append("\\section{Transcription Task Full Results}")
    lines.append("\\label{app:transcription-full}")
    lines.append("")
    lines.append(
        f"Table~\\ref{{tab:transcription-full}} reports per-model transcription task metrics across all {n_samples} samples "
        "(multi-run mean over three runs)."
    )
    lines.append(
        "PP denotes whether postprocessing was enabled (\\checkmark) or disabled "
        "($\\times$). Bold entries indicate the best value per column."
    )
    lines.append("")
    lines.append("\\begin{table}[htbp]")
    lines.append("  \\centering")
    lines.append(
        "  \\caption{Transcription task full results "
        f"($n={n_samples}$ per run, 3 runs averaged per model). "
        "PP = postprocessing (\\checkmark\\ on, $\\times$\\ off). "
        "$\\downarrow$ lower is better; $\\uparrow$ higher is better.}"
    )
    lines.append("  \\label{tab:transcription-full}")
    lines.append("  \\setlength{\\tabcolsep}{6pt}")
    lines.append("  \\renewcommand{\\arraystretch}{1.15}")
    lines.append("  \\small")
    lines.append("  \\begin{tabular}{llrrrrrrr}")
    lines.append("    \\toprule")
    lines.append(
        "    \\textbf{Model} & \\textbf{PP}"
        " & \\textbf{CER}$\\downarrow$"
        " & \\textbf{WER}$\\downarrow$"
        " & \\textbf{NED}$\\downarrow$"
        " & \\textbf{Prec.}$\\uparrow$"
        " & \\textbf{Recall}$\\uparrow$"
        " & \\textbf{BoW F1}$\\uparrow$"
        " & \\textbf{Exact}$\\uparrow$ \\\\"
    )
    lines.append("    \\midrule")

    for i, row in enumerate(rows):
        model = display_model_name(row["model"])
        pp = "\\checkmark" if row["pp"] else "$\\times$"
        vals = []
        for m in metrics:
            s = format_float(row["metrics"][m], 4)
            if is_close(row["metrics"][m], best_vals[m]):
                s = f"\\textbf{{{s}}}"
            vals.append(s)
        lines.append(
            f"    {model} & {pp} & {vals[0]} & {vals[1]} & {vals[2]} & {vals[3]} & {vals[4]} & {vals[5]} & {vals[6]} \\\\"
        )
        if i % 2 == 1 and i != len(rows) - 1:
            lines.append("    \\midrule")

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def render_full_table_structure(rows: List[dict], metrics: List[str], n_samples: int) -> str:
    lower_better = {"HW CER", "HW WER", "HW NED"}
    best_vals = best_by_metric(rows, metrics, lower_better)

    lines = []
    lines.append("\\section{Structure Extraction Task Full Results}")
    lines.append("\\label{app:structure-full}")
    lines.append("")
    lines.append(
        f"Table~\\ref{{tab:structure-full}} reports per-model structure extraction task metrics across all {n_samples} samples "
        "(multi-run mean over three runs)."
    )
    lines.append(
        "Weighted denotes the sample-level mean of $0.5\\times(\\mathrm{Y/N\\ Acc} + \\mathrm{HW\\ Sim})$, "
        "where $\\mathrm{HW\\ Sim}=\\max(0, 1-\\mathrm{CER})$ is computed per sample before averaging, "
        "so it is not recoverable by plugging the aggregate HW CER column into the formula."
    )
    lines.append(
        "HW CER, HW WER, and HW NED are effectively unchanged between PP-on and PP-off rows for each model, "
        "consistent with the postprocessing switch mainly affecting the Y/N extraction branch."
    )
    lines.append("")
    lines.append("\\begin{table}[htbp]")
    lines.append("  \\centering")
    lines.append(
        "  \\caption{Structure extraction task full results "
        f"($n={n_samples}$ per run, 3 runs averaged per model). "
        "PP = postprocessing (\\checkmark\\ on, $\\times$\\ off). "
        "$\\downarrow$ lower is better; $\\uparrow$ higher is better.}"
    )
    lines.append("  \\label{tab:structure-full}")
    lines.append("  \\setlength{\\tabcolsep}{10pt}")
    lines.append("  \\renewcommand{\\arraystretch}{1.15}")
    lines.append("  \\small")
    lines.append("  \\begin{tabular}{llrrrrr}")
    lines.append("    \\toprule")
    lines.append(
        "    \\textbf{Model} & \\textbf{PP}"
        " & \\textbf{Weighted}$\\uparrow$"
        " & \\textbf{Y/N Acc}$\\uparrow$"
        " & \\textbf{HW CER}$\\downarrow$"
        " & \\textbf{HW WER}$\\downarrow$"
        " & \\textbf{HW NED}$\\downarrow$ \\\\"
    )
    lines.append("    \\midrule")

    for i, row in enumerate(rows):
        model = display_model_name(row["model"])
        pp = "\\checkmark" if row["pp"] else "$\\times$"
        vals = []
        for m in metrics:
            s = format_float(row["metrics"][m], 4)
            if is_close(row["metrics"][m], best_vals[m]):
                s = f"\\textbf{{{s}}}"
            vals.append(s)
        lines.append(f"    {model} & {pp} & {vals[0]} & {vals[1]} & {vals[2]} & {vals[3]} & {vals[4]} \\\\")
        if i % 2 == 1 and i != len(rows) - 1:
            lines.append("    \\midrule")

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def collect_checkbox_rows() -> Tuple[List[dict], float, int, int, int]:
    gt_path = "data/sample_gt_v2.json"
    gt_rows = load_json(gt_path)
    pos = 0
    neg = 0
    for row in gt_rows:
        yn = row.get("yn_options", {})
        if not isinstance(yn, dict):
            continue
        for value in yn.values():
            label = str(value).strip().upper()
            if label == "Y":
                pos += 1
            elif label == "N":
                neg += 1
    total = pos + neg
    positive_rate = (pos / total) if total else 0.0

    grouped: Dict[str, List[dict]] = defaultdict(list)
    for path in sorted(glob.glob("results/multirun/preds_v2_*__run*.json")):
        name = os.path.basename(path)
        model_with_run = name[len("preds_v2_"):-len(".json")]
        model_id, _run = model_with_run.rsplit("__run", 1)
        if str(base_model_id(model_id)).startswith("dummy"):
            continue
        preds = load_json(path)
        report = OCREvaluatorV2(gt_path, enable_postprocess=True).evaluate_results(preds)
        yn_stats = report["field_analysis"]["yn_options"]
        grouped[base_model_id(model_id)].append(
            {
                "acc": float(report.get("avg_yn_acc", 0.0) or 0.0),
                "prec": float(report.get("yn_positive_precision", 0.0) or 0.0),
                "recall": float(report.get("yn_positive_recall", 0.0) or 0.0),
                "f1": float(report.get("yn_positive_f1", 0.0) or 0.0),
                "bal_acc": float(report.get("yn_balanced_accuracy", 0.0) or 0.0),
                "tp": float(yn_stats.get("tp", 0.0) or 0.0),
                "tn": float(yn_stats.get("tn", 0.0) or 0.0),
                "fp": float(yn_stats.get("fp", 0.0) or 0.0),
                "fn": float(yn_stats.get("fn", 0.0) or 0.0),
                "miss_pos": float(yn_stats.get("missing_pos", 0.0) or 0.0),
                "miss_neg": float(yn_stats.get("missing_neg", 0.0) or 0.0),
                "weighted": float(report.get("avg_weighted_score", 0.0) or 0.0),
            }
        )

    out = []
    for model, vals in grouped.items():
        def avg(key: str) -> float:
            return float(np.mean([v[key] for v in vals])) if vals else 0.0
        out.append(
            {
                "model": model,
                "weighted": avg("weighted"),
                "acc": avg("acc"),
                "prec": avg("prec"),
                "recall": avg("recall"),
                "f1": avg("f1"),
                "bal_acc": avg("bal_acc"),
                "tp": avg("tp"),
                "tn": avg("tn"),
                "fp": avg("fp"),
                "fn": avg("fn"),
                "miss_pos": avg("miss_pos"),
                "miss_neg": avg("miss_neg"),
            }
        )
    out.sort(key=lambda r: (-r["weighted"], r["model"]))
    return out, positive_rate, total, pos, neg


def render_checkbox_metrics_table(rows: List[dict], positive_rate: float, total_fields: int, pos_fields: int, neg_fields: int) -> str:
    best_recall = max((r["recall"] for r in rows), default=0.0)
    best_f1 = max((r["f1"] for r in rows), default=0.0)
    best_bal_acc = max((r["bal_acc"] for r in rows), default=0.0)

    lines = []
    lines.append("\\section{Checkbox Positive-Class Metrics}")
    lines.append("\\label{app:checkbox-positive}")
    lines.append("")
    lines.append(
        f"Because positive checkbox marks are sparse in the structured split ({positive_rate*100:.1f}\\% positive; "
        f"{total_fields} checkbox fields in total, with {pos_fields} positive and {neg_fields} negative fields per run), "
        "Table~\\ref{tab:checkbox-positive} reports field-level confusion statistics and positive-class metrics for postprocessed runs."
    )
    lines.append(
        "We separate unmatched or invalid checkbox outputs into Missing$+$ (ground-truth Y) and Missing$-$ (ground-truth N); "
        "under this definition, Y/N accuracy is exactly $(TP+TN)/(TP+TN+FP+FN+\\text{Missing}^++\\text{Missing}^-)$."
    )
    lines.append("")
    lines.append("\\begin{table}[htbp]")
    lines.append("  \\centering")
    lines.append(
        "  \\caption{Field-level checkbox confusion statistics and positive-class metrics on the structure extraction task (PP on, 3 runs averaged per model). Counts are run means, so they need not be integers.}"
    )
    lines.append("  \\label{tab:checkbox-positive}")
    lines.append("  \\small")
    lines.append("  \\setlength{\\tabcolsep}{5pt}")
    lines.append("  \\begin{tabular}{lrrrrrrrrrr}")
    lines.append("    \\toprule")
    lines.append(
        "    \\textbf{Model} & \\textbf{TP} & \\textbf{TN} & \\textbf{FP} & \\textbf{FN} & \\textbf{Miss$+$} & \\textbf{Miss$-$} & \\textbf{Y/N Acc}$\\uparrow$ & \\textbf{Y Recall}$\\uparrow$ & \\textbf{Y F1}$\\uparrow$ & \\textbf{Bal. Acc}$\\uparrow$ \\\\"
    )
    lines.append("    \\midrule")
    for row in rows:
        tp = format_float(row["tp"], 1)
        tn = format_float(row["tn"], 1)
        fp = format_float(row["fp"], 1)
        fn = format_float(row["fn"], 1)
        miss_pos = format_float(row["miss_pos"], 1)
        miss_neg = format_float(row["miss_neg"], 1)
        acc = format_float(row["acc"], 4)
        recall = format_float(row["recall"], 4)
        f1 = format_float(row["f1"], 4)
        bal_acc = format_float(row["bal_acc"], 4)
        if is_close(row["recall"], best_recall):
            recall = f"\\textbf{{{recall}}}"
        if is_close(row["f1"], best_f1):
            f1 = f"\\textbf{{{f1}}}"
        if is_close(row["bal_acc"], best_bal_acc):
            bal_acc = f"\\textbf{{{bal_acc}}}"
        lines.append(
            f"    {display_model_name(row['model'])} & {tp} & {tn} & {fp} & {fn} & {miss_pos} & {miss_neg} & {acc} & {recall} & {f1} & {bal_acc} \\\\"
        )
    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def reconstruct_model_file_means(v_key: str) -> Dict[str, Dict[str, float]]:
    dist_rows = load_json(f"results/multirun/distribution_{v_key}.json")
    gt_path = "data/sample_gt_v1.json" if v_key == "v1" else "data/sample_gt_v2.json"
    gt_rows = load_json(gt_path)
    file_order = [x["file_name"] for x in gt_rows]

    grouped_scores: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    for row in dist_rows:
        model = str(row.get("Base Model") or "")
        run = int(row.get("Run") or 0)
        score = float(row.get("Score") or 0.0)
        if not model or run <= 0:
            continue
        grouped_scores[(model, run)].append(score)

    model_file_scores: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    n = len(file_order)
    for (model, _run), scores in grouped_scores.items():
        if len(scores) != n:
            raise ValueError(f"Unexpected sample count for {model}: got {len(scores)}, expected {n}")
        for idx, score in enumerate(scores):
            model_file_scores[model][file_order[idx]].append(score)

    model_file_means: Dict[str, Dict[str, float]] = {}
    for model, f_map in model_file_scores.items():
        model_file_means[model] = {fn: float(np.mean(vals)) for fn, vals in f_map.items() if vals}
    return model_file_means


def compute_per_type_for_best_model(best_model: str) -> Tuple[List[dict], int]:
    evaluator = OCREvaluator("data/sample_gt_v1.json", normalize=True)
    paths = sorted(glob.glob(f"results/multirun/preds_v1_{best_model}__run*.json"))
    if not paths:
        raise FileNotFoundError(f"No multirun prediction files for model: {best_model}")

    category_stats = {k: {"cer": [], "bow_f1": []} for k in KNOWN_V1_CATEGORIES}
    for path in paths:
        preds = load_json(path)
        report = evaluator.evaluate_results(preds)
        for d in report.get("details", []):
            fn = d.get("file_name")
            if not fn:
                continue
            cat = v1_category(fn)
            if cat not in category_stats:
                continue
            category_stats[cat]["cer"].append(float(d.get("cer", 0.0) or 0.0))
            category_stats[cat]["bow_f1"].append(float(d.get("bow_f1", 0.0) or 0.0))

    gt = load_json("data/sample_gt_v1.json")
    sample_counts = defaultdict(int)
    for row in gt:
        sample_counts[v1_category(row["file_name"])] += 1

    rows = []
    for cat in KNOWN_V1_CATEGORIES:
        cer_vals = category_stats[cat]["cer"]
        f1_vals = category_stats[cat]["bow_f1"]
        if not cer_vals:
            continue
        rows.append(
            {
                "category": CATEGORY_LABELS[cat],
                "prefix": cat,
                "n": int(sample_counts[cat]),
                "cer": float(np.mean(cer_vals)),
                "bow_f1": float(np.mean(f1_vals)),
            }
        )
    return rows, len(paths)


def render_per_type_table(rows: List[dict], best_model: str, n_runs: int) -> str:
    lines = []
    lines.append("\\section{Per-Form-Type Performance Analysis}")
    lines.append("\\label{app:per-type}")
    lines.append("")
    lines.append(
        "The transcription task contains seven form-style categories "
        "(hw, pn, cf, hd, rl, ess, numeric). "
        f"Table~\\ref{{tab:per-type}} reports mean CER and BoW~F1 for the best-performing model "
        f"(\\texttt{{{display_model_name(best_model)}}}, PP on), averaged over {n_runs} multiruns."
    )
    lines.append("")
    lines.append("\\begin{table}[htbp]")
    lines.append("  \\centering")
    lines.append(
        f"  \\caption{{Per-form-type transcription results for \\texttt{{{display_model_name(best_model)}}} (PP on, {n_runs} runs averaged). "
        "$n$ = unique sample count per category.}"
    )
    lines.append("  \\label{tab:per-type}")
    lines.append("  \\small")
    lines.append("  \\begin{tabular}{lcrr}")
    lines.append("    \\toprule")
    lines.append("    \\textbf{Category} & \\textbf{Prefix} & \\textbf{CER}$\\downarrow$ & \\textbf{BoW F1}$\\uparrow$ \\\\")
    lines.append("    \\midrule")
    for r in rows:
        lines.append(f"    {r['category']} & {r['prefix']} & {r['cer']:.4f} & {r['bow_f1']:.4f} \\\\")
    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def build_edge_case_data(model_file_means_v1: Dict[str, Dict[str, float]], on_rows_v1: List[dict], best_model: str):
    all_files = sorted(set().union(*[set(v.keys()) for v in model_file_means_v1.values()]))
    hard = []
    for fn in all_files:
        vals = [f_map[fn] for f_map in model_file_means_v1.values() if fn in f_map]
        if not vals:
            continue
        hard.append(
            {
                "file": fn,
                "count_ge_02": int(sum(1 for x in vals if x >= 0.2)),
                "model_count": len(vals),
                "mean_cer": float(np.mean(vals)),
            }
        )

    hard_sorted = sorted(hard, key=lambda x: (-x["count_ge_02"], -x["mean_cer"], x["file"]))
    hard_top3 = hard_sorted[:3]

    v1_on_mean = {r["model"]: r["metrics"]["Avg CER"] for r in on_rows_v1 if r["pp"]}
    preferred = [best_model, "gemini-3.1-pro-preview", "gpt-5-mini"]
    rep_models = []
    for m in preferred:
        if m in model_file_means_v1 and m not in rep_models:
            rep_models.append(m)
    if len(rep_models) < 3:
        for m, _ in sorted(v1_on_mean.items(), key=lambda kv: kv[1]):
            if m in model_file_means_v1 and m not in rep_models:
                rep_models.append(m)
            if len(rep_models) >= 3:
                break

    edge_rows = []
    for m in rep_models[:3]:
        row = {
            "model": m,
            "mean": float(v1_on_mean.get(m, 0.0)),
        }
        for h in hard_top3:
            row[h["file"]] = float(model_file_means_v1[m].get(h["file"], 0.0))
        edge_rows.append(row)

    catastrophic = None
    if "gpt-5" in model_file_means_v1 and best_model in model_file_means_v1:
        common = sorted(set(model_file_means_v1["gpt-5"].keys()) & set(model_file_means_v1[best_model].keys()))
        if common:
            gaps = []
            for fn in common:
                gpt5_cer = model_file_means_v1["gpt-5"][fn]
                best_cer = model_file_means_v1[best_model][fn]
                gaps.append((fn, gpt5_cer, best_cer, gpt5_cer - best_cer))
            fn, gpt5_cer, best_cer, gap = max(gaps, key=lambda x: x[3])
            catastrophic = {
                "file": fn,
                "gpt5_cer": float(gpt5_cer),
                "best_model": best_model,
                "best_cer": float(best_cer),
                "gap": float(gap),
                "ratio": float(gpt5_cer / max(best_cer, 1e-12)),
            }

    return hard_top3, edge_rows, catastrophic


def render_edge_cases(hard_top3: List[dict], edge_rows: List[dict], catastrophic: dict | None) -> str:
    hard_files = [h["file"] for h in hard_top3]
    min_count = min(h["count_ge_02"] for h in hard_top3)
    max_count = max(h["count_ge_02"] for h in hard_top3)
    model_n = hard_top3[0]["model_count"] if hard_top3 else 0

    lines = []
    lines.append("\\section{Challenging Edge Cases}")
    lines.append("\\label{app:edge-cases}")
    lines.append("")
    lines.append(
        "Our per-sample transcription analysis surfaces two failure modes: "
        "(i) systematically hard samples across model families, and "
        "(ii) model-specific catastrophic degradations on otherwise tractable inputs."
    )
    lines.append("")
    lines.append("\\paragraph{Systematically hard samples.}")
    lines.append(
        "Using cross-model mean CER from multirun outputs, "
        f"we identify \\texttt{{{hard_files[0]}}}, \\texttt{{{hard_files[1]}}}, and \\texttt{{{hard_files[2]}}} "
        f"as the hardest samples. These files show CER $\\ge 0.2$ for {min_count}--{max_count} of {model_n} models, "
        "suggesting dataset-level difficulty rather than isolated decoding errors."
    )
    lines.append("Table~\\ref{tab:edge-scores} reports representative model CERs on these samples and each model's transcription-task mean CER.")
    lines.append("")
    lines.append("\\begin{table}[htbp]")
    lines.append("  \\centering")
    lines.append(
        "  \\caption{CER on systematically hard transcription samples (post-processing on) and each model's transcription-task mean CER.}"
    )
    lines.append("  \\label{tab:edge-scores}")
    lines.append("  \\small")
    lines.append("  \\begin{tabular}{lcccc}")
    lines.append("    \\toprule")
    lines.append(
        f"    \\textbf{{Model}} & \\textbf{{{hard_files[0].replace('.png', '')}}} & \\textbf{{{hard_files[1].replace('.png', '')}}} & \\textbf{{{hard_files[2].replace('.png', '')}}} & \\textbf{{Mean CER}} \\\\"
    )
    lines.append("    \\midrule")
    for r in edge_rows:
        lines.append(
            f"    {display_model_name(r['model'])} & {r[hard_files[0]]:.4f} & {r[hard_files[1]]:.4f} & {r[hard_files[2]]:.4f} & {r['mean']:.4f} \\\\"
        )
    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    lines.append("\\paragraph{Model-specific catastrophic failure.}")
    if catastrophic:
        lines.append(
            f"A distinct failure mode appears on \\texttt{{{catastrophic['file']}}}: \\texttt{{{display_model_name(catastrophic['best_model'])}}} "
            f"achieves CER {catastrophic['best_cer']:.4f}, while \\texttt{{gpt-5}} reaches CER {catastrophic['gpt5_cer']:.4f} "
            f"(absolute gap = {catastrophic['gap']:.4f}, ratio = {catastrophic['ratio']:.1f}$\\times$). "
            "This large gap is consistent with occasional output instability in gpt-5 on this benchmark and motivates caution when interpreting its aggregate rank."
        )
    else:
        lines.append("No stable model-specific catastrophic failure pattern was detected under the current multirun setting.")

    return "\n".join(lines)


def compute_pairwise_structure(model_file_means_v2: Dict[str, Dict[str, float]], on_rows_v2: List[dict]) -> Tuple[List[dict], int]:
    mean_map = {r["model"]: r["metrics"]["Weighted Score"] for r in on_rows_v2 if r["pp"]}
    model_order = [m for m, _ in sorted(mean_map.items(), key=lambda kv: kv[1], reverse=True)]

    rows = []
    n_samples = 0
    for m1, m2 in combinations(model_order, 2):
        files = sorted(set(model_file_means_v2[m1].keys()) & set(model_file_means_v2[m2].keys()))
        s1 = np.array([model_file_means_v2[m1][f] for f in files], dtype=float)
        s2 = np.array([model_file_means_v2[m2][f] for f in files], dtype=float)
        n_samples = len(files)

        if len(files) >= 2:
            test = stats.ttest_rel(s1, s2)
            p_value = float(test.pvalue) if np.isfinite(test.pvalue) else 1.0
        else:
            p_value = 1.0

        m1_mean = float(np.mean(s1)) if len(s1) else 0.0
        m2_mean = float(np.mean(s2)) if len(s2) else 0.0

        if p_value < 0.05:
            winner = display_model_name(m1) if m1_mean > m2_mean else display_model_name(m2)
        else:
            winner = "---"

        rows.append(
            {
                "model1": m1,
                "model2": m2,
                "m1": m1_mean,
                "m2": m2_mean,
                "p": p_value,
                "winner": winner,
            }
        )

    return rows, n_samples


def format_p_value(p: float) -> str:
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def render_pairwise_table(rows: List[dict], n_samples: int) -> str:
    lines = []
    lines.append("\\section{Full Pairwise Significance Tests}")
    lines.append("\\label{sec:appendix_pairwise}")
    lines.append("")
    lines.append(
        f"Table~\\ref{{tab:pairwise_full}} reports paired $t$-tests ($n={n_samples}$) "
        "for all model pairs on structure extraction task weighted score. "
        "Significance threshold: $\\alpha = 0.05$."
    )
    lines.append("")
    lines.append("\\begin{table}[p]")
    lines.append("\\centering")
    lines.append(
        f"\\caption{{Full pairwise paired $t$-test results on structure extraction task weighted score ($n={n_samples}$).}}"
    )
    lines.append("\\label{tab:pairwise_full}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{llcccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Model 1} & \\textbf{Model 2} & \\textbf{M1} & \\textbf{M2} & \\textbf{$p$-value} & \\textbf{Winner} \\\\")
    lines.append("\\midrule")

    current_m1 = None
    for r in rows:
        m1 = display_model_name(r["model1"])
        m2 = display_model_name(r["model2"])
        if current_m1 is not None and m1 != current_m1:
            lines.append("\\midrule")
        current_m1 = m1
        lines.append(
            f"{m1} & {m2} & {r['m1']:.3f} & {r['m2']:.3f} & {format_p_value(r['p'])} & {r['winner']} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def render_prompts_section() -> str:
    v1_prompt = """Please perform OCR on this image. Return the recognized text exactly as it appears.
CRITICAL RULES FOR FAIR EVALUATION:
1. Do NOT add any numbering (like '1.', '2.') unless they are printed in the image.
2. Do NOT add any markdown formatting, code blocks (```), tables, or bold text.
3. Do NOT fix typos, grammar, punctuation, or 'improve' the text. Output exactly what is seen.
4. Do NOT add any extra punctuation or spaces to align text. Preserve the original layout roughly via line breaks.
5. Do NOT try to format the output as a list if it's not a clear list in the image.
6. For selection options (like Yes/No or checkboxes), represent the marked choice as 'Y' or 'N'.
7. Do NOT include any introductory or concluding remarks. Start the output directly with the recognized text."""

    v2_prompt = """Perform OCR with two goals: (1) extract all handwritten text, and
(2) detect Y/N selections for any options or checkboxes, regardless of form format.

Return ONLY a JSON object with this structure:
{
  "handwriting_text": "...",
  "yn_options": { "Label 1": "Y/N", "Label 2": "Y/N" }
}

RULES:
1. handwriting_text: include ONLY handwritten content. Preserve line breaks when possible. If none, use an empty string.
2. yn_options: use the visible option/question label as the key (any language). Mark 'Y' only if the Y/Yes option is explicitly ticked/circled/checked. If N/No is marked, or neither is marked, use 'N'.
3. Do NOT add extra text outside the JSON. Do NOT use markdown/code blocks."""

    lines = []
    lines.append("\\section{Prompts for OCR Tasks}")
    lines.append("\\label{app:prompts}")
    lines.append("")
    lines.append(
        "For reproducibility, we report the centralized prompts for the two OCR settings in MedDocBench. "
        "The transcription task uses a general-purpose text transcription prompt, whereas the structure extraction task "
        "uses a structured prompt that separates handwritten text from binary Y/N option detection."
    )
    lines.append("")
    lines.append("\\paragraph{Transcription task: standard text-based OCR (general purpose).}")
    lines.append("\\begin{verbatim}")
    lines.append(v1_prompt)
    lines.append("\\end{verbatim}")
    lines.append("")
    lines.append("\\paragraph{Structure extraction task: simple handwriting + Y/N extraction (format-agnostic).}")
    lines.append("\\begin{verbatim}")
    lines.append(v2_prompt)
    lines.append("\\end{verbatim}")
    return "\n".join(lines)


def generate_latex(output_path: str):
    v1_rows, v1_metrics, v1_n = collect_full_rows("v1")
    v2_rows, v2_metrics, v2_n = collect_full_rows("v2")
    checkbox_rows, checkbox_positive_rate, checkbox_total_fields, checkbox_pos_fields, checkbox_neg_fields = collect_checkbox_rows()

    best_model = min([r for r in v1_rows if r["pp"]], key=lambda r: r["metrics"]["Avg CER"])["model"]

    model_file_means_v1 = reconstruct_model_file_means("v1")
    model_file_means_v2 = reconstruct_model_file_means("v2")

    per_type_rows, n_runs = compute_per_type_for_best_model(best_model)
    hard_top3, edge_rows, catastrophic = build_edge_case_data(
        model_file_means_v1=model_file_means_v1,
        on_rows_v1=[r for r in v1_rows if r["pp"]],
        best_model=best_model,
    )
    pairwise_rows, pair_n = compute_pairwise_structure(model_file_means_v2, [r for r in v2_rows if r["pp"]])

    sections = []
    sections.append("%% Auto-generated from multirun artifacts")
    sections.append(render_prompts_section())
    sections.append("\\clearpage")
    sections.append(render_full_table_transcription(v1_rows, v1_metrics, v1_n))
    sections.append("\\clearpage")
    sections.append(render_full_table_structure(v2_rows, v2_metrics, v2_n))
    sections.append("\\clearpage")
    sections.append(
        render_checkbox_metrics_table(
            checkbox_rows,
            checkbox_positive_rate,
            checkbox_total_fields,
            checkbox_pos_fields,
            checkbox_neg_fields,
        )
    )
    sections.append("\\clearpage")
    sections.append(render_per_type_table(per_type_rows, best_model, n_runs))
    sections.append("\\clearpage")
    sections.append(render_edge_cases(hard_top3, edge_rows, catastrophic))
    sections.append("\\clearpage")
    sections.append(render_pairwise_table(pairwise_rows, pair_n))

    content = "\n\n".join(sections) + "\n"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Wrote: {output_path}")
    print(f"Transcription best model: {display_model_name(best_model)}")
    print(f"Transcription samples per run: {v1_n}; Structure samples per run: {v2_n}")
    print(f"Hard samples: {[h['file'] for h in hard_top3]}")
    if catastrophic:
        print(
            "Catastrophic case:",
            catastrophic["file"],
            f"gpt-5={catastrophic['gpt5_cer']:.4f}",
            f"{display_model_name(catastrophic['best_model'])}={catastrophic['best_cer']:.4f}",
        )


def main():
    parser = argparse.ArgumentParser(description="Generate appendix LaTeX from multirun artifacts.")
    parser.add_argument("--output", default="/tmp/multirun_appendix.tex", help="Output .tex path")
    args = parser.parse_args()
    generate_latex(args.output)


if __name__ == "__main__":
    main()
