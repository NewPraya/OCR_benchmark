import streamlit as st
import json
import os
import io
import pandas as pd
from PIL import Image
import glob
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns
from main import run_benchmark
from evaluators.evaluator import OCREvaluator
from evaluators.evaluator_v2 import OCREvaluatorV2
from evaluators.statistical_tests import (
    compare_models,
    batch_compare_models
)
from utils.dataset_splits import load_splits, get_split_for_version

st.set_page_config(page_title="OCR Benchmark Dashboard", layout="wide")

st.title("📊 OCR Benchmark Dashboard")

# --- Sidebar: Configuration ---
st.sidebar.header("Run Benchmark")
eval_version = st.sidebar.radio("Evaluation Mode", ["v1 (Text)", "v2 (Simple)"], index=0)
v_key = "v1" if "v1" in eval_version else "v2"

model_type = st.sidebar.selectbox("Select Model Type", ["gemini", "qwen", "openai", "ollama", "dummy"])
default_ids = (
    "gemini-2.0-flash-exp"
    if model_type == "gemini"
    else ("llama3.2-vision" if model_type == "ollama" else ("gpt-4.1-mini" if model_type == "openai" else "gpt-4o"))
)
model_ids_input = st.sidebar.text_area("Model IDs (one per line)", value=default_ids)
disable_postprocess = st.sidebar.checkbox("Disable post-processing (ablation)", value=False)
run_btn = st.sidebar.button("🚀 Run Benchmark")

if run_btn:
    model_ids = [m.strip() for m in model_ids_input.split("\n") if m.strip()]
    with st.spinner(f"Running {v_key.upper()} benchmark for {len(model_ids)} models..."):
        run_benchmark(model_type, model_ids, eval_version=v_key, postprocess=(not disable_postprocess))
    st.success("Benchmark completed!")

# --- Load Results ---
def _resolve_gt_path(v_key):
    if v_key == "v2":
        return "data/sample_gt_v2.json"
    return "data/sample_gt_v1.json" if os.path.exists("data/sample_gt_v1.json") else "data/sample_gt.json"

def _file_signature(path):
    if not os.path.exists(path):
        return (path, 0.0, 0)
    stat = os.stat(path)
    return (path, stat.st_mtime, stat.st_size)

def _result_file_signatures(v_key):
    result_files = sorted(glob.glob(f"results/preds_{v_key}_*.json"))
    return tuple(_file_signature(path) for path in result_files)

def _report_file_signatures(v_key):
    report_files = sorted(glob.glob(f"results/report_{v_key}_*.json"))
    return tuple(_file_signature(path) for path in report_files)

def _save_report_file(v_key, model_id, report):
    os.makedirs("results", exist_ok=True)
    safe_model_id = model_id.replace("/", "_")
    report_path = f"results/report_{v_key}_{safe_model_id}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

def _resolve_pred_path(v_key, model_id):
    candidates = [
        f"results/preds_{v_key}_{model_id}.json",
        f"results/preds_{v_key}_{model_id.replace('/', '_')}.json",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[1]

def _base_model_id(model_id: str) -> str:
    return model_id[:-9] if model_id.endswith("__no_post") else model_id

def _report_summary_row(v_key, model_id, report, postprocess_default=True):
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
            "Weighted Score": round(report.get('avg_weighted_score', 0.0), 4),
            "Y/N Acc": round(report.get('avg_yn_acc', 0.0), 4),
            "HW CER": round(report.get('avg_handwriting_cer', 0.0), 4),
            "HW WER": round(report.get('avg_handwriting_wer', 0.0), 4),
            "HW NED": round(report.get('avg_handwriting_ned', 0.0), 4),
        }
    return {
        **common,
        "Avg CER": round(report.get('average_cer', 0.0), 4),
        "Avg WER": round(report.get('average_wer', 0.0), 4),
        "Avg NED": round(report.get('average_ned', 0.0), 4),
        "Precision": round(report.get('average_precision', 0.0), 4),
        "Recall": round(report.get('average_recall', 0.0), 4),
        "BoW F1": round(report.get('average_bow_f1', 0.0), 4),
        "Exact Match": round(report.get('exact_match_accuracy', 0.0), 4),
    }

def _model_family_from_id(model_id: str) -> str:
    base = _base_model_id(model_id).lower()
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

@st.cache_data(show_spinner=False)
def _load_all_results_cached(v_key, gt_sig, split_sig, result_sigs, report_sigs):
    results = []
    result_files = [sig[0] for sig in result_sigs]
    report_files = [sig[0] for sig in report_sigs]
    gt_path = gt_sig[0]
    if not os.path.exists(gt_path):
        return [], {}

    if v_key == "v2":
        evaluator = OCREvaluatorV2(gt_path)
    else:
        evaluator = OCREvaluator(gt_path)
    split_set = get_split_for_version(load_splits(), v_key)

    # Fast path: load precomputed reports if available.
    if report_files:
        for f in report_files:
            try:
                with open(f, 'r') as j:
                    report = json.load(j)
                model_id = report.get("model_id") or os.path.basename(f).replace(f"report_{v_key}_", "").replace(".json", "")
                results.append(_report_summary_row(v_key, model_id, report, postprocess_default=True))
            except Exception:
                continue
        gt_dict = {k: v for k, v in evaluator.gt_dict.items() if (not split_set or k in split_set)}
        return results, gt_dict
    
    # Fallback path for legacy result files without precomputed report.
    for f in result_files:
        model_id = os.path.basename(f).replace(f"preds_{v_key}_", "").replace(".json", "")
        with open(f, 'r') as j:
            try:
                predictions = json.load(j)
                if not predictions: continue
                report = evaluator.evaluate_results(predictions)
                # Backfill summary report for faster future dashboard loads.
                _save_report_file(v_key, model_id, report)
                results.append(_report_summary_row(v_key, model_id, report, postprocess_default=True))
            except Exception:
                continue
    gt_dict = {k: v for k, v in evaluator.gt_dict.items() if (not split_set or k in split_set)}
    return results, gt_dict

def load_all_results(v_key):
    gt_path = _resolve_gt_path(v_key)
    gt_sig = _file_signature(gt_path)
    split_sig = _file_signature("data/dataset_split.json")
    result_sigs = _result_file_signatures(v_key)
    report_sigs = _report_file_signatures(v_key)
    return _load_all_results_cached(v_key, gt_sig, split_sig, result_sigs, report_sigs)

results_data, gt_dict = load_all_results(v_key)

# --- Helper Functions ---
@st.cache_data(show_spinner=False)
def _load_full_results_cached(v_key, gt_sig, split_sig, result_sigs, report_sigs):
    """Load full evaluation results including details (cached)."""
    result_files = [sig[0] for sig in result_sigs]
    report_files = [sig[0] for sig in report_sigs]
    gt_path = gt_sig[0]

    if not os.path.exists(gt_path):
        return {}

    full_results = {}

    # Fast path: direct report loading.
    if report_files:
        for f in report_files:
            try:
                with open(f, 'r') as j:
                    report = json.load(j)
                model_id = report.get("model_id") or os.path.basename(f).replace(f"report_{v_key}_", "").replace(".json", "")
                full_results[model_id] = report
            except Exception:
                continue
        return full_results

    # Fallback path for legacy preds files.
    evaluator = OCREvaluatorV2(gt_path) if v_key == "v2" else OCREvaluator(gt_path)
    for f in result_files:
        model_id = os.path.basename(f).replace(f"preds_{v_key}_", "").replace(".json", "")
        with open(f, 'r') as j:
            try:
                predictions = json.load(j)
                if predictions:
                    report = evaluator.evaluate_results(predictions)
                    # Backfill report for faster future dashboard loads.
                    _save_report_file(v_key, model_id, report)
                    full_results[model_id] = report
            except Exception:
                continue

    return full_results

def load_full_results(v_key):
    gt_path = _resolve_gt_path(v_key)
    gt_sig = _file_signature(gt_path)
    split_sig = _file_signature("data/dataset_split.json")
    result_sigs = _result_file_signatures(v_key)
    report_sigs = _report_file_signatures(v_key)
    return _load_full_results_cached(v_key, gt_sig, split_sig, result_sigs, report_sigs)

@st.cache_data(show_spinner=False)
def _load_predictions_file_cached(path, file_sig):
    with open(path, 'r') as f:
        return json.load(f)

def export_to_latex(df, caption="OCR Benchmark Results"):
    """Export dataframe to LaTeX table format."""
    latex_str = df.to_latex(index=False, float_format="%.4f", caption=caption, label="tab:results")
    return latex_str


def _detail_quality_score(v_key, detail):
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


def _build_case_finder_df(v_key, full_results, model_a, model_b):
    """Build a per-sample dataframe to find hard/ambiguous/disagreement cases."""
    rep_a = full_results.get(model_a, {})
    rep_b = full_results.get(model_b, {})
    det_a = {d.get("file_name"): d for d in rep_a.get("details", []) if isinstance(d, dict)}
    det_b = {d.get("file_name"): d for d in rep_b.get("details", []) if isinstance(d, dict)}
    common = sorted(set(det_a.keys()) & set(det_b.keys()))
    rows = []
    for fname in common:
        a = det_a[fname]
        b = det_b[fname]
        qa = _detail_quality_score(v_key, a)
        qb = _detail_quality_score(v_key, b)
        row = {
            "file_name": fname,
            f"{model_a}_quality": qa,
            f"{model_b}_quality": qb,
            "avg_quality": (qa + qb) / 2.0,
            "abs_gap": abs(qa - qb),
            "winner": model_a if qa > qb else (model_b if qb > qa else "tie"),
        }
        if v_key == "v2":
            row.update({
                f"{model_a}_yn_acc": float(a.get("yn_acc", 0.0) or 0.0),
                f"{model_b}_yn_acc": float(b.get("yn_acc", 0.0) or 0.0),
                f"{model_a}_hw_cer": float(a.get("handwriting_cer", 0.0) or 0.0),
                f"{model_b}_hw_cer": float(b.get("handwriting_cer", 0.0) or 0.0),
            })
        else:
            row.update({
                f"{model_a}_cer": float(a.get("cer", 1.0) or 1.0),
                f"{model_b}_cer": float(b.get("cer", 1.0) or 1.0),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def build_export_bundle(v_key, df, caption, full_results=None, stats_metric=None):
    """
    Build a zip bundle for paper writing.
    Includes leaderboard exports, raw reports, and optional pairwise stats.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Leaderboard artifacts
        zf.writestr(f"{v_key}/leaderboard_{v_key}.csv", df.to_csv(index=False))
        zf.writestr(f"{v_key}/leaderboard_{v_key}.json", df.to_json(orient='records', indent=2))
        zf.writestr(f"{v_key}/leaderboard_{v_key}.tex", export_to_latex(df, caption=caption))

        # Raw report files
        for report_path in sorted(glob.glob(f"results/report_{v_key}_*.json")):
            try:
                with open(report_path, "rb") as f:
                    zf.writestr(f"{v_key}/reports/{os.path.basename(report_path)}", f.read())
            except Exception:
                continue

        # Optional pairwise statistics summary
        if full_results and len(full_results) >= 2 and stats_metric:
            try:
                comparisons = batch_compare_models(full_results, stats_metric, use_parametric=True)
                rows = []
                for comp in comparisons:
                    rows.append({
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
                        "winner": comp.get("winner")
                    })
                if rows:
                    stats_df = pd.DataFrame(rows)
                    zf.writestr(f"{v_key}/stats/pairwise_stats_{stats_metric}.csv", stats_df.to_csv(index=False))
                    zf.writestr(f"{v_key}/stats/pairwise_stats_{stats_metric}.json", json.dumps(rows, indent=2))
            except Exception:
                pass

        # Minimal manifest
        manifest = {
            "version": v_key,
            "rows": int(len(df)),
            "columns": list(df.columns),
            "caption": caption
        }
        zf.writestr(f"{v_key}/manifest.json", json.dumps(manifest, indent=2))

    buf.seek(0)
    return buf.getvalue()


def _stats_metric_options(v_key: str):
    if v_key == "v2":
        return ["weighted_score", "yn_acc", "handwriting_cer", "handwriting_wer", "handwriting_ned"], "weighted_score"
    return ["cer", "wer", "ned", "bow_f1", "exact_match"], "cer"


def _format_ci(ci_pair):
    low = ci_pair[0] if isinstance(ci_pair, (list, tuple)) and len(ci_pair) > 0 else None
    high = ci_pair[1] if isinstance(ci_pair, (list, tuple)) and len(ci_pair) > 1 else None
    if low is None or high is None:
        return "N/A"
    return f"[{low:.4f}, {high:.4f}]"


def _build_pairwise_table(comparisons):
    rows = []
    for comp in comparisons:
        test = comp.get("statistical_test", {})
        rows.append({
            "Model 1": comp.get("model1_id", "N/A"),
            "Model 2": comp.get("model2_id", "N/A"),
            "M1 Mean": f"{comp.get('model1', {}).get('mean', 0.0):.4f}",
            "M1 95% CI": _format_ci(comp.get("model1", {}).get("ci_95")),
            "M2 Mean": f"{comp.get('model2', {}).get('mean', 0.0):.4f}",
            "M2 95% CI": _format_ci(comp.get("model2", {}).get("ci_95")),
            "p-value": f"{(test.get('p_value') if test.get('p_value') is not None else 0.0):.6f}",
            "Significant": "✓" if test.get("significant", False) else "✗",
            "Winner": comp.get("winner", "N/A"),
        })
    return pd.DataFrame(rows)

# --- Tabs for better organization ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Leaderboard", "🔍 Detailed View", "📈 Statistical Analysis", "📤 Export"])

# --- Main UI: Overview ---
with tab1:
    if results_data:
        st.header(f"📈 Leaderboard ({v_key.upper()})")
        df = pd.DataFrame(results_data)
        sort_col = "Weighted Score" if v_key == "v2" else "Avg CER"
        ascending = False if v_key == "v2" else True
        df = df.sort_values(sort_col, ascending=ascending)
        st.dataframe(df, width='stretch')
        
        # Show summary statistics
        st.subheader("Summary Statistics")
        st.write(df.describe())

        # Ablation grouped bar chart: same model with postprocess ON/OFF side-by-side
        if "Postprocess" in df.columns:
            chart_df = df.copy()
            chart_df["Base Model"] = chart_df["Model ID"].apply(_base_model_id)
            paired = chart_df.groupby("Base Model")["Postprocess"].nunique()
            paired_models = paired[paired >= 2].index.tolist()

            if paired_models:
                st.subheader("Postprocess Ablation (Same Model Paired)")

                metric_candidates = [
                    c for c in chart_df.columns
                    if c not in {"Model ID", "Base Model", "Postprocess", "Samples", "Target", "Processed", "Failed", "Failed Rate"}
                    and pd.api.types.is_numeric_dtype(chart_df[c])
                ]
                default_metric = "Weighted Score" if v_key == "v2" else "Avg CER"
                metric_idx = metric_candidates.index(default_metric) if default_metric in metric_candidates else 0
                selected_metric = st.selectbox(
                    "Metric for paired bar chart",
                    metric_candidates,
                    index=metric_idx,
                    key=f"paired_bar_metric_{v_key}"
                )

                lower_is_better = selected_metric in {
                    "Avg CER", "Avg WER", "Avg NED", "HW CER", "HW WER", "HW NED"
                }

                paired_df = chart_df[chart_df["Base Model"].isin(paired_models)]
                pivot = paired_df.pivot_table(
                    index="Base Model",
                    columns="Postprocess",
                    values=selected_metric,
                    aggfunc="mean"
                )

                # Sort by postprocess-ON if available, else by OFF.
                sort_col_bool = True if True in pivot.columns else (False if False in pivot.columns else pivot.columns[0])
                pivot = pivot.sort_values(sort_col_bool, ascending=lower_is_better)

                fig, ax = plt.subplots(figsize=(max(10, len(pivot) * 0.9), 5))
                x = list(range(len(pivot.index)))
                width = 0.38

                if False in pivot.columns:
                    vals_off = pivot[False].tolist()
                    ax.bar([i - width / 2 for i in x], vals_off, width=width, label="Postprocess OFF")
                if True in pivot.columns:
                    vals_on = pivot[True].tolist()
                    ax.bar([i + width / 2 for i in x], vals_on, width=width, label="Postprocess ON")

                ax.set_xticks(x)
                ax.set_xticklabels(pivot.index.tolist(), rotation=35, ha='right')
                ax.set_ylabel(selected_metric)
                ax.set_title(f"Paired Ablation by Model ({selected_metric})")
                ax.grid(True, axis='y', alpha=0.3)
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)

                delta_rows = []
                for model_name, row in pivot.iterrows():
                    on_val = row.get(True, None)
                    off_val = row.get(False, None)
                    if on_val is None or off_val is None:
                        continue
                    delta_rows.append({
                        "Base Model": model_name,
                        "Postprocess ON": round(float(on_val), 4),
                        "Postprocess OFF": round(float(off_val), 4),
                        "Delta (ON - OFF)": round(float(on_val - off_val), 4)
                    })
                if delta_rows:
                    st.caption("Delta > 0 means ON is numerically higher; for error metrics (CER/WER/NED), lower is better.")
                    st.dataframe(pd.DataFrame(delta_rows), width='stretch')

        st.divider()
        st.subheader("Custom Chart Builder")
        st.caption("Filter model scope and metrics, then plot freely for paper figures.")

        custom_df = df.copy()
        custom_df["Base Model"] = custom_df["Model ID"].apply(_base_model_id)
        custom_df["Model Family"] = custom_df["Model ID"].apply(_model_family_from_id)
        custom_df["Postprocess Label"] = custom_df["Postprocess"].apply(lambda x: "ON" if bool(x) else "OFF")

        family_options = sorted(custom_df["Model Family"].unique().tolist())
        selected_families = st.multiselect(
            "Model families",
            options=family_options,
            default=family_options
        )

        state_options = ["ON", "OFF"] if "Postprocess" in custom_df.columns else ["ON"]
        selected_states = st.multiselect(
            "Postprocess states",
            options=state_options,
            default=state_options
        )

        family_filtered_df = custom_df[
            custom_df["Model Family"].isin(selected_families)
            & custom_df["Postprocess Label"].isin(selected_states)
        ]

        model_options = family_filtered_df["Model ID"].tolist()
        selected_models = st.multiselect(
            "Specific models (optional, choose subset)",
            options=model_options,
            default=model_options
        )
        plot_df = family_filtered_df[family_filtered_df["Model ID"].isin(selected_models)]

        metric_options = [
            c for c in plot_df.columns
            if c not in {"Model ID", "Base Model", "Model Family", "Postprocess", "Postprocess Label", "Samples", "Target", "Processed", "Failed", "Failed Rate"}
            and pd.api.types.is_numeric_dtype(plot_df[c])
        ]
        default_metric = "Weighted Score" if v_key == "v2" else "Avg CER"
        selected_metrics = st.multiselect(
            "Metrics",
            options=metric_options,
            default=[default_metric] if default_metric in metric_options else metric_options[:1]
        )

        chart_type = st.selectbox("Chart type", ["Grouped Bar", "Line"])

        if plot_df.empty:
            st.info("No rows match current filters.")
        elif not selected_metrics:
            st.info("Select at least one metric.")
        else:
            fig, axes = plt.subplots(
                nrows=len(selected_metrics),
                ncols=1,
                figsize=(max(10, len(plot_df) * 0.65), max(4, 3.5 * len(selected_metrics))),
                squeeze=False
            )

            for idx, metric in enumerate(selected_metrics):
                ax = axes[idx][0]
                metric_plot_df = plot_df.sort_values("Model ID").copy()

                if chart_type == "Grouped Bar":
                    sns.barplot(
                        data=metric_plot_df,
                        x="Model ID",
                        y=metric,
                        hue="Postprocess Label",
                        errorbar=None,
                        ax=ax
                    )
                else:
                    sns.lineplot(
                        data=metric_plot_df,
                        x="Model ID",
                        y=metric,
                        hue="Postprocess Label",
                        marker="o",
                        ax=ax
                    )

                ax.set_title(metric)
                ax.set_xlabel("Model ID")
                ax.set_ylabel(metric)
                ax.tick_params(axis="x", rotation=35)
                ax.grid(True, axis="y", alpha=0.25)
                ax.legend(title="Postprocess")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            export_cols = ["Model ID", "Base Model", "Model Family", "Postprocess Label"] + selected_metrics
            export_df = plot_df[export_cols].sort_values(["Model Family", "Base Model", "Postprocess Label"])
            st.dataframe(export_df, width='stretch')
            st.download_button(
                label="Download filtered chart data (CSV)",
                data=export_df.to_csv(index=False),
                file_name=f"custom_plot_data_{v_key}.csv",
                mime="text/csv"
            )
    else:
        st.info(f"No {v_key.upper()} results found. Run a benchmark to see data.")

# --- Tab 2: Detailed View ---
with tab2:
    if results_data:
        st.header("🔍 Detailed Comparison")

        detail_key = f"detail_selected_image_{v_key}"
        image_options = list(gt_dict.keys())
        if image_options:
            if detail_key not in st.session_state or st.session_state[detail_key] not in image_options:
                st.session_state[detail_key] = image_options[0]
        selected_image = st.selectbox("Select Image to Inspect", image_options, key=detail_key)
        if v_key == "v2":
            gt_path = "data/sample_gt_v2.json"
        else:
            gt_path = "data/sample_gt_v1.json" if os.path.exists("data/sample_gt_v1.json") else "data/sample_gt.json"
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Original Image")
            img_path = os.path.join("data", selected_image)
            if os.path.exists(img_path):
                st.image(img_path, width='stretch')
            else:
                st.error(f"Image not found: {img_path}")
                
            st.subheader("Ground Truth")
            if v_key == "v2":
                st.json(gt_dict[selected_image])
            else:
                st.text_area("GT Text", gt_dict[selected_image], height=300, disabled=True)

        with col2:
            st.subheader("Model Predictions")
            selected_models = st.multiselect("Select Models to Compare", df["Model ID"].tolist(), default=df["Model ID"].tolist()[:2])
            
            for mid in selected_models:
                res_file = _resolve_pred_path(v_key, mid)
                if os.path.exists(res_file):
                    preds = _load_predictions_file_cached(res_file, _file_signature(res_file))
                    pred_item = next((p for p in preds if p['file_name'] == selected_image), None)
                    if pred_item:
                        with st.expander(f"Model: {mid}", expanded=True):
                            if v_key == "v2":
                                try:
                                    pred_json = json.loads(pred_item['prediction']) if isinstance(pred_item['prediction'], str) else pred_item['prediction']
                                    st.json(pred_json)
                                except Exception:
                                    st.error("Failed to parse JSON")
                                    st.text(pred_item['prediction'])
                            else:
                                st.text_area(f"Prediction ({mid})", pred_item['prediction'], height=200, key=f"pred_{mid}")
                else:
                    st.warning(f"No results for model {mid}")
        
        if v_key == "v2" and selected_models:
            st.divider()
            st.subheader("Y/N Comparison (GT-first)")
            evaluator = OCREvaluatorV2(gt_path)
            gt_item = gt_dict.get(selected_image, {})
            gt_yn = gt_item.get("yn_options", {})
            rows = []
            for gt_label, gt_value in gt_yn.items():
                row = {
                    "Question (GT Label)": gt_label,
                    "GT": evaluator._normalize_yn(gt_value) or str(gt_value)
                }
                for mid in selected_models:
                    res_file = _resolve_pred_path(v_key, mid)
                    pred_value = "-"
                    if os.path.exists(res_file):
                        preds = _load_predictions_file_cached(res_file, _file_signature(res_file))
                        pred_item = next((p for p in preds if p['file_name'] == selected_image), None)
                        if pred_item:
                            try:
                                pred_json = json.loads(pred_item['prediction']) if isinstance(pred_item['prediction'], str) else pred_item['prediction']
                            except Exception:
                                pred_json = {}
                            pred_yn_raw = evaluator._extract_yn_options_from_pred(pred_json)
                            pred_entries = [(k, v, evaluator._normalize_key_variants(k)) for k, v in pred_yn_raw.items()]
                            match_val, match_type, _ = evaluator._find_pred_value(gt_label, pred_entries)
                            norm_val = evaluator._normalize_yn(match_val)
                            if norm_val:
                                pred_value = norm_val
                            elif match_val is not None:
                                pred_value = str(match_val)
                            if match_type and pred_value != "-" and match_type not in ("exact_full", "exact_ascii"):
                                pred_value = f"{pred_value} ({match_type})"
                    row[mid] = pred_value
                rows.append(row)
            if rows:
                yn_df = pd.DataFrame(rows)
                st.dataframe(yn_df, width='stretch')

        if v_key == "v2":
            st.divider()
            st.subheader("Y/N Error Breakdown")
            full_results = load_full_results(v_key)
            if full_results:
                default_model = selected_models[0] if selected_models else list(full_results.keys())[0]
                stats_model = st.selectbox("Model for Y/N stats", list(full_results.keys()), index=list(full_results.keys()).index(default_model))
                top_n = st.slider("Show lowest-accuracy questions", min_value=5, max_value=50, value=20, step=5)
                stats = full_results.get(stats_model, {}).get("yn_question_stats", [])
                if stats:
                    df_stats = pd.DataFrame(stats)
                    df_stats = df_stats.head(top_n)
                    if "match_types" in df_stats.columns:
                        df_stats["match_types"] = df_stats["match_types"].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else x)
                    st.dataframe(df_stats, width='stretch')
                else:
                    st.info("No per-question stats available for this model.")
            else:
                st.info("No results available for Y/N breakdown.")

        st.divider()
        st.subheader("Case Finder (Hard / Borderline / Disagreement)")
        st.caption("Use this to quickly locate partial failures instead of manually searching image-by-image.")
        full_results_for_cases = load_full_results(v_key)
        if len(full_results_for_cases) >= 2:
            available_models = list(full_results_for_cases.keys())
            default_a = selected_models[0] if selected_models else available_models[0]
            default_b = selected_models[1] if (selected_models and len(selected_models) > 1) else (
                available_models[1] if len(available_models) > 1 else available_models[0]
            )
            col_cf1, col_cf2 = st.columns(2)
            with col_cf1:
                case_model_a = st.selectbox("Case Finder Model A", available_models, index=available_models.index(default_a), key=f"cf_model_a_{v_key}")
            with col_cf2:
                idx_b = available_models.index(default_b) if default_b in available_models else min(1, len(available_models) - 1)
                case_model_b = st.selectbox("Case Finder Model B", available_models, index=idx_b, key=f"cf_model_b_{v_key}")

            if case_model_a == case_model_b:
                st.info("Choose two different models to mine disagreement and boundary cases.")
            else:
                cases_df = _build_case_finder_df(v_key, full_results_for_cases, case_model_a, case_model_b)
                if not cases_df.empty:
                    mode = st.selectbox(
                        "Ranking mode",
                        ["Hard cases (both models low)", "Borderline cases (mid quality)", "Disagreement cases (large gap)"],
                        key=f"cf_mode_{v_key}"
                    )
                    top_n = st.slider("Rows to show", min_value=5, max_value=50, value=20, step=5, key=f"cf_topn_{v_key}")
                    gap_threshold = st.slider("Minimum quality gap", min_value=0.0, max_value=1.0, value=0.1, step=0.01, key=f"cf_gap_{v_key}")

                    filt_df = cases_df[cases_df["abs_gap"] >= gap_threshold].copy()
                    if mode == "Hard cases (both models low)":
                        ranked_df = filt_df.sort_values(["avg_quality", "abs_gap"], ascending=[True, False]).head(top_n)
                    elif mode == "Borderline cases (mid quality)":
                        filt_df["borderline_distance"] = (filt_df["avg_quality"] - 0.5).abs()
                        ranked_df = filt_df.sort_values(["borderline_distance", "abs_gap"], ascending=[True, False]).head(top_n)
                    else:
                        ranked_df = filt_df.sort_values(["abs_gap", "avg_quality"], ascending=[False, True]).head(top_n)

                    st.dataframe(ranked_df, width='stretch')
                    st.caption("Tip: copy `file_name` from this table, then use the image selector above to inspect details.")
                else:
                    st.info("No common per-sample details found for the selected model pair.")
        else:
            st.info("Need at least two models with full report details to use Case Finder.")
    else:
        st.info(f"No {v_key.upper()} results found. Run a benchmark to see data.")

# --- Tab 3: Statistical Analysis ---
with tab3:
    if results_data and len(results_data) >= 2:
        st.header("📈 Statistical Analysis")
        
        full_results = load_full_results(v_key)
        
        if len(full_results) >= 2:
            metric_options, default_metric = _stats_metric_options(v_key)
            metric_idx = metric_options.index(default_metric) if default_metric in metric_options else 0
            selected_metric = st.selectbox("Select Metric for Analysis", metric_options, index=metric_idx)
            
            # Model comparison selector
            model_ids = list(full_results.keys())
            col1, col2 = st.columns(2)
            with col1:
                model1 = st.selectbox("Model 1", model_ids, index=0)
            with col2:
                model2 = st.selectbox("Model 2", model_ids, index=min(1, len(model_ids)-1))
            
            use_parametric = st.checkbox("Use parametric test (t-test)", value=True, 
                                        help="Uncheck to use non-parametric Wilcoxon test")
            
            if st.button("🔬 Run Statistical Comparison"):
                if model1 == model2:
                    st.warning("Please select two different models to compare.")
                else:
                    with st.spinner("Running statistical tests..."):
                        comparison = compare_models(
                            full_results[model1],
                            full_results[model2],
                            metric_name=selected_metric,
                            use_parametric=use_parametric
                        )
                    
                    st.subheader("Comparison Results")
                    
                    # Display means with confidence intervals
                    col1, col2, col3 = st.columns(3)
                    m1_ci = comparison.get("model1", {}).get("ci_95", (None, None))
                    m2_ci = comparison.get("model2", {}).get("ci_95", (None, None))
                    with col1:
                        st.metric(
                            f"{model1}",
                            f"{comparison['model1']['mean']:.4f}",
                            help=f"95% CI: {_format_ci(m1_ci)}"
                        )
                    with col2:
                        st.metric(
                            f"{model2}",
                            f"{comparison['model2']['mean']:.4f}",
                            help=f"95% CI: {_format_ci(m2_ci)}"
                        )
                    with col3:
                        winner = comparison.get('winner', 'N/A')
                        st.metric("Winner", winner)
                    
                    # Statistical test results
                    st.subheader("Statistical Test")
                    test_result = comparison['statistical_test']
                    
                    test_df = pd.DataFrame([{
                        'Test': test_result.get('test', 'N/A'),
                        'p-value': f"{test_result.get('p_value', 0):.6f}",
                        'Significant': '✓' if test_result.get('significant', False) else '✗',
                        'Interpretation': test_result.get('interpretation', 'N/A')
                    }])
                    st.dataframe(test_df, width='stretch')
                    
                    # Visualization
                    st.subheader("Score Distribution")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    
                    # Box plot
                    data_to_plot = [comparison['model1']['scores'], comparison['model2']['scores']]
                    ax.boxplot(data_to_plot, labels=[model1, model2])
                    ax.set_ylabel(selected_metric)
                    ax.set_title(f"{selected_metric} Distribution Comparison")
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                    plt.close()
            
            # Batch comparison
            st.divider()
            st.subheader("Pairwise Comparisons (All Models)")
            
            if st.button("🔬 Run All Pairwise Comparisons"):
                with st.spinner("Running pairwise comparisons..."):
                    comparisons = batch_compare_models(full_results, selected_metric, use_parametric)

                comp_df = _build_pairwise_table(comparisons)
                st.dataframe(comp_df, width='stretch')
                st.download_button(
                    label="Download pairwise stats (CSV)",
                    data=comp_df.to_csv(index=False),
                    file_name=f"pairwise_stats_{v_key}_{selected_metric}.csv",
                    mime="text/csv",
                    key=f"download_pairwise_{v_key}_{selected_metric}"
                )
        else:
            st.info("Need at least 2 models with results for statistical comparison.")
    else:
        st.info("Need at least 2 models with results for statistical analysis.")

# --- Tab 4: Export ---
with tab4:
    if results_data:
        st.header("📤 Export Results")
        
        # LaTeX export
        st.subheader("LaTeX Table")
        df = pd.DataFrame(results_data)
        
        caption = st.text_input("Table Caption", f"OCR Benchmark Results ({v_key.upper()} Mode)")
        
        if st.button("Generate LaTeX"):
            latex_code = export_to_latex(df, caption=caption)
            st.code(latex_code, language='latex')
            st.download_button(
                label="Download LaTeX",
                data=latex_code,
                file_name=f"ocr_results_{v_key}.tex",
                mime="text/plain"
            )
        
        # CSV export
        st.subheader("CSV Export")
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"ocr_results_{v_key}.csv",
            mime="text/csv"
        )
        
        # JSON export
        st.subheader("JSON Export")
        json_str = df.to_json(orient='records', indent=2)
        st.download_button(
            label="Download JSON",
            data=json_str,
            file_name=f"ocr_results_{v_key}.json",
            mime="application/json"
        )

        st.divider()
        st.subheader("One-click Paper Bundle")
        st.caption("Package leaderboard exports + raw reports + pairwise stats into one zip file.")
        bundle_metric = "weighted_score" if v_key == "v2" else "cer"
        bundle_caption = st.text_input(
            "Bundle table caption",
            f"OCR Benchmark Results ({v_key.upper()} Mode)",
            key=f"bundle_caption_{v_key}"
        )
        if st.button("Prepare ZIP Bundle", key=f"prepare_zip_{v_key}"):
            with st.spinner("Building zip bundle..."):
                full_results = load_full_results(v_key)
                bundle_bytes = build_export_bundle(
                    v_key=v_key,
                    df=df,
                    caption=bundle_caption,
                    full_results=full_results,
                    stats_metric=bundle_metric
                )
                st.session_state[f"bundle_bytes_{v_key}"] = bundle_bytes
            st.success("Bundle prepared. Click download below.")

        if st.session_state.get(f"bundle_bytes_{v_key}"):
            st.download_button(
                label="Download Paper Bundle (.zip)",
                data=st.session_state[f"bundle_bytes_{v_key}"],
                file_name=f"meddocbench_{v_key}_paper_bundle.zip",
                mime="application/zip",
                key=f"download_zip_{v_key}"
            )
    else:
        st.info("No results available for export.")

# --- Tools ---
st.sidebar.divider()
st.sidebar.header("Tools")
if st.sidebar.button("🔄 Sync MD to GT (Legacy V2 Labels)"):
    from utils.sync_to_gt import sync_to_gt
    sync_to_gt()
    st.sidebar.success("Synced labels to V2 GT!")
    st.rerun()
