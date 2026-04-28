import json
import os

import streamlit as st

from evaluators.evaluator import OCREvaluator
from evaluators.evaluator_v2 import OCREvaluatorV2
from utils.dataset_splits import load_splits, get_split_for_version

from dashboard.utils import (
    file_signature,
    report_file_signatures,
    report_summary_row,
    resolve_gt_path,
    result_file_signatures,
    save_report_file,
)


@st.cache_data(show_spinner=False)
def _load_all_results_cached(v_key, gt_sig, split_sig, result_sigs, report_sigs):
    results = []
    result_files = [sig[0] for sig in result_sigs]
    report_files = [sig[0] for sig in report_sigs]
    gt_path = gt_sig[0]
    if not os.path.exists(gt_path):
        return [], {}

    evaluator = OCREvaluatorV2(gt_path) if v_key == "v2" else OCREvaluator(gt_path)
    split_set = get_split_for_version(load_splits(), v_key)

    if report_files:
        for path in report_files:
            try:
                with open(path, "r") as j:
                    report = json.load(j)
                model_id = report.get("model_id") or os.path.basename(path).replace(f"report_{v_key}_", "").replace(".json", "")
                results.append(report_summary_row(v_key, model_id, report, postprocess_default=True))
            except Exception:
                continue
        gt_dict = {k: v for k, v in evaluator.gt_dict.items() if (not split_set or k in split_set)}
        return results, gt_dict

    for path in result_files:
        model_id = os.path.basename(path).replace(f"preds_{v_key}_", "").replace(".json", "")
        with open(path, "r") as j:
            try:
                predictions = json.load(j)
                if not predictions:
                    continue
                report = evaluator.evaluate_results(predictions)
                save_report_file(v_key, model_id, report)
                results.append(report_summary_row(v_key, model_id, report, postprocess_default=True))
            except Exception:
                continue

    gt_dict = {k: v for k, v in evaluator.gt_dict.items() if (not split_set or k in split_set)}
    return results, gt_dict


def load_all_results(v_key):
    gt_path = resolve_gt_path(v_key)
    gt_sig = file_signature(gt_path)
    split_sig = file_signature("data/dataset_split.json")
    result_sigs = result_file_signatures(v_key)
    report_sigs = report_file_signatures(v_key)
    return _load_all_results_cached(v_key, gt_sig, split_sig, result_sigs, report_sigs)


@st.cache_data(show_spinner=False)
def _load_full_results_cached(v_key, gt_sig, split_sig, result_sigs, report_sigs):
    result_files = [sig[0] for sig in result_sigs]
    report_files = [sig[0] for sig in report_sigs]
    gt_path = gt_sig[0]

    if not os.path.exists(gt_path):
        return {}

    full_results = {}

    if report_files:
        for path in report_files:
            try:
                with open(path, "r") as j:
                    report = json.load(j)
                model_id = report.get("model_id") or os.path.basename(path).replace(f"report_{v_key}_", "").replace(".json", "")
                full_results[model_id] = report
            except Exception:
                continue
        return full_results

    evaluator = OCREvaluatorV2(gt_path) if v_key == "v2" else OCREvaluator(gt_path)
    for path in result_files:
        model_id = os.path.basename(path).replace(f"preds_{v_key}_", "").replace(".json", "")
        with open(path, "r") as j:
            try:
                predictions = json.load(j)
                if predictions:
                    report = evaluator.evaluate_results(predictions)
                    save_report_file(v_key, model_id, report)
                    full_results[model_id] = report
            except Exception:
                continue

    return full_results


def load_full_results(v_key):
    gt_path = resolve_gt_path(v_key)
    gt_sig = file_signature(gt_path)
    split_sig = file_signature("data/dataset_split.json")
    result_sigs = result_file_signatures(v_key)
    report_sigs = report_file_signatures(v_key)
    return _load_full_results_cached(v_key, gt_sig, split_sig, result_sigs, report_sigs)


@st.cache_data(show_spinner=False)
def load_predictions_file_cached(path, file_sig):
    with open(path, "r") as f:
        return json.load(f)
