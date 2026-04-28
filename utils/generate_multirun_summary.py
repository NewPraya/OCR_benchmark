"""
Precompute multi-run leaderboard artifacts for fast dashboard loading.

Usage:
  python utils/generate_multirun_summary.py --version all --write-csv
"""

import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import pandas as pd

# Allow direct execution via `python utils/generate_multirun_summary.py`.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from evaluators.evaluator import OCREvaluator
from evaluators.evaluator_v2 import OCREvaluatorV2
from dashboard.utils import base_model_id, model_family_from_id, report_summary_row, resolve_gt_path


MULTIRUN_PATTERN = re.compile(r"^preds_(v1|v2)_(.+)__run(\d+)\.json$")
EXCLUDED_DISPLAY_MODELS = {"gemini-3-pro"}

V1_METRICS = [
    "Avg CER",
    "Avg WER",
    "Avg NED",
    "Precision",
    "Recall",
    "BoW F1",
    "Exact Match",
    "Failed Rate",
]

V2_METRICS = [
    "Weighted Score",
    "Y/N Acc",
    "Y Pos Prec",
    "Y Pos Recall",
    "Y Pos F1",
    "Y Bal Acc",
    "HW CER",
    "HW WER",
    "HW NED",
    "Failed Rate",
]

BASE_COLUMNS = [
    "Model ID",
    "Postprocess",
    "Runs",
    "Samples",
    "Target",
    "Processed",
    "Failed",
    "Failed Rate",
]


def _parse_multirun_file(path: str):
    name = os.path.basename(path)
    match = MULTIRUN_PATTERN.match(name)
    if not match:
        return None
    version, model_id, run_str = match.groups()
    return version, model_id, int(run_str)


def _postprocess_enabled(model_id: str) -> bool:
    return not model_id.endswith("__no_post")


def _display_model_name(model_id: str) -> str:
    return base_model_id(model_id).replace("-preview", "")


def _is_excluded_model(model_id: str) -> bool:
    return _display_model_name(model_id) in EXCLUDED_DISPLAY_MODELS


def _failed_stats(predictions: List[dict]) -> Tuple[int, int, float]:
    entries = [p for p in predictions if isinstance(p, dict)]
    total = len(entries)
    failed = sum(1 for p in entries if bool(p.get("failed")))
    failed_rate = (failed / total) if total else 0.0
    return total, failed, failed_rate


def _metric_cols(v_key: str) -> List[str]:
    return V2_METRICS if v_key == "v2" else V1_METRICS


def _collect_files(v_key: str) -> List[Tuple[str, str, int]]:
    paths = sorted(glob.glob(f"results/multirun/preds_{v_key}_*__run*.json"))
    parsed = []
    for path in paths:
        p = _parse_multirun_file(path)
        if p:
            _, model_id, run_idx = p
            parsed.append((path, model_id, run_idx))
    parsed.sort(key=lambda x: (x[1], x[2]))
    return parsed


def _evaluate_per_run(v_key: str, include_off_from_same_preds: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    files = _collect_files(v_key)
    if not files:
        return pd.DataFrame(), pd.DataFrame()

    gt_path = resolve_gt_path(v_key)
    if not os.path.exists(gt_path):
        return pd.DataFrame(), pd.DataFrame()

    evaluator_cache: Dict[bool, object] = {}
    rows = []
    distribution_rows = []
    detail_key = "cer" if v_key == "v1" else "weighted_score"

    for path, model_id, run_idx in files:
        try:
            with open(path, "r") as f:
                predictions = json.load(f)
            if not isinstance(predictions, list) or not predictions:
                continue
        except Exception:
            continue

        postprocess = _postprocess_enabled(model_id)
        eval_modes = [postprocess]
        if include_off_from_same_preds and postprocess:
            eval_modes = [True, False]

        total, failed, failed_rate = _failed_stats(predictions)
        for mode in eval_modes:
            if mode not in evaluator_cache:
                if v_key == "v2":
                    evaluator_cache[mode] = OCREvaluatorV2(gt_path, enable_postprocess=mode)
                else:
                    evaluator_cache[mode] = OCREvaluator(gt_path, normalize=mode)
            evaluator = evaluator_cache[mode]

            report = evaluator.evaluate_results(predictions)
            row = report_summary_row(v_key, model_id, report, postprocess_default=mode)
            row["Samples"] = total
            row["Target"] = total
            row["Processed"] = total
            row["Failed"] = failed
            row["Failed Rate"] = round(failed_rate, 4)
            row["Run"] = run_idx
            rows.append(row)

            # Distribution chart uses ON only, precomputed here to avoid real-time heavy evaluation in dashboard.
            if mode:
                base_id = base_model_id(model_id)
                if str(base_id).startswith("dummy") or _is_excluded_model(base_id):
                    continue
                for d in report.get("details", []):
                    if not isinstance(d, dict):
                        continue
                    val = d.get(detail_key)
                    if val is None:
                        continue
                    try:
                        distribution_rows.append(
                            {
                                "Base Model": base_id,
                                "Display Model": _display_model_name(model_id),
                                "Family": model_family_from_id(model_id),
                                "Score": float(val),
                                "Run": int(run_idx),
                            }
                        )
                    except Exception:
                        continue

    if not rows:
        return pd.DataFrame(), pd.DataFrame()

    per_run_df = pd.DataFrame(rows).sort_values(["Model ID", "Run"]).reset_index(drop=True)
    distribution_df = pd.DataFrame(distribution_rows)
    return per_run_df, distribution_df


def _aggregate_leaderboard(v_key: str, per_run_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if per_run_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    metric_cols = _metric_cols(v_key)
    agg_rows = []

    for (model_id, postprocess), group_df in per_run_df.groupby(["Model ID", "Postprocess"], dropna=False):
        row = {
            "Model ID": model_id,
            "Postprocess": bool(postprocess),
            "Runs": int(group_df["Run"].nunique()),
            "Samples": int(round(group_df["Samples"].mean())),
            "Target": int(round(group_df["Target"].mean())) if "Target" in group_df else 0,
            "Processed": int(round(group_df["Processed"].mean())) if "Processed" in group_df else 0,
            "Failed": int(round(group_df["Failed"].mean())) if "Failed" in group_df else 0,
            "Failed Rate": round(float(group_df["Failed Rate"].mean()), 4) if "Failed Rate" in group_df else 0.0,
        }

        for metric in metric_cols:
            if metric in group_df.columns:
                row[metric] = round(float(group_df[metric].mean()), 4)
                row[f"{metric} Std"] = round(float(group_df[metric].std(ddof=0)), 4)

        agg_rows.append(row)

    leaderboard_df = pd.DataFrame(agg_rows)

    if leaderboard_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    if v_key == "v2" and "Weighted Score" in leaderboard_df.columns:
        leaderboard_df = leaderboard_df.sort_values("Weighted Score", ascending=False)
    if v_key == "v1" and "Avg CER" in leaderboard_df.columns:
        leaderboard_df = leaderboard_df.sort_values("Avg CER", ascending=True)

    std_cols = [c for c in leaderboard_df.columns if c.endswith(" Std")]
    main_metric_cols = [c for c in metric_cols if c in leaderboard_df.columns]

    visible_cols = BASE_COLUMNS + [c for c in main_metric_cols if c not in BASE_COLUMNS]
    visible_cols = [c for c in visible_cols if c in leaderboard_df.columns]

    std_df = leaderboard_df[["Model ID", "Postprocess"] + std_cols].copy() if std_cols else pd.DataFrame()
    leaderboard_df = leaderboard_df[visible_cols].reset_index(drop=True)

    return leaderboard_df, std_df


def _write_json(path: str, rows: List[dict]):
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)


def generate(v_key: str, write_csv: bool = False, include_off_from_same_preds: bool = True):
    os.makedirs("results/multirun", exist_ok=True)

    per_run_df, distribution_df = _evaluate_per_run(v_key, include_off_from_same_preds=include_off_from_same_preds)
    leaderboard_df, std_df = _aggregate_leaderboard(v_key, per_run_df)

    per_run_json = f"results/multirun/per_run_{v_key}.json"
    leaderboard_json = f"results/multirun/leaderboard_{v_key}.json"
    std_json = f"results/multirun/leaderboard_std_{v_key}.json"
    distribution_json = f"results/multirun/distribution_{v_key}.json"
    meta_json = f"results/multirun/summary_meta_{v_key}.json"

    _write_json(per_run_json, per_run_df.to_dict(orient="records"))
    _write_json(leaderboard_json, leaderboard_df.to_dict(orient="records"))
    _write_json(std_json, std_df.to_dict(orient="records"))
    _write_json(distribution_json, distribution_df.to_dict(orient="records"))

    meta = {
        "version": v_key,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "per_run_rows": int(len(per_run_df)),
        "leaderboard_rows": int(len(leaderboard_df)),
        "std_rows": int(len(std_df)),
        "distribution_rows": int(len(distribution_df)),
        "source_files": len(_collect_files(v_key)),
    }
    _write_json(meta_json, meta)

    if write_csv:
        per_run_df.to_csv(f"results/multirun/per_run_{v_key}.csv", index=False)
        leaderboard_df.to_csv(f"results/multirun/leaderboard_{v_key}.csv", index=False)
        std_df.to_csv(f"results/multirun/leaderboard_std_{v_key}.csv", index=False)
        distribution_df.to_csv(f"results/multirun/distribution_{v_key}.csv", index=False)

    print(f"[{v_key}] per-run rows: {len(per_run_df)}")
    print(f"[{v_key}] leaderboard rows: {len(leaderboard_df)}")
    print(f"[{v_key}] distribution rows: {len(distribution_df)}")
    print(f"[{v_key}] wrote: {leaderboard_json}, {per_run_json}, {std_json}, {distribution_json}, {meta_json}")


def main():
    parser = argparse.ArgumentParser(description="Precompute multi-run summaries for dashboard")
    parser.add_argument("--version", choices=["v1", "v2", "all"], default="all")
    parser.add_argument("--write-csv", action="store_true", help="Also write CSV outputs")
    parser.add_argument(
        "--include-off-from-same-preds",
        action="store_true",
        default=True,
        help="Also evaluate postprocess OFF using the same multirun prediction files.",
    )
    parser.add_argument(
        "--no-include-off-from-same-preds",
        dest="include_off_from_same_preds",
        action="store_false",
        help="Disable OFF evaluation from the same multirun prediction files.",
    )
    args = parser.parse_args()

    versions = ["v1", "v2"] if args.version == "all" else [args.version]
    for v in versions:
        generate(
            v,
            write_csv=args.write_csv,
            include_off_from_same_preds=args.include_off_from_same_preds,
        )


if __name__ == "__main__":
    main()
