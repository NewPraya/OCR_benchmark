import streamlit as st
import json
import os
import pandas as pd
from PIL import Image
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from main import run_benchmark
from evaluators.evaluator import OCREvaluator
from evaluators.evaluator_v2 import OCREvaluatorV2
from evaluators.statistical_tests import (
    bootstrap_confidence_interval,
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
run_btn = st.sidebar.button("🚀 Run Benchmark")

if run_btn:
    model_ids = [m.strip() for m in model_ids_input.split("\n") if m.strip()]
    with st.spinner(f"Running {v_key.upper()} benchmark for {len(model_ids)} models..."):
        run_benchmark(model_type, model_ids, eval_version=v_key)
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
                if v_key == "v2":
                    results.append({
                        "Model ID": model_id,
                        "Weighted Score": round(report.get('avg_weighted_score', 0.0), 4),
                        "Y/N Acc": round(report.get('avg_yn_acc', 0.0), 4),
                        "HW CER": round(report.get('avg_handwriting_cer', 0.0), 4),
                        "HW WER": round(report.get('avg_handwriting_wer', 0.0), 4),
                        "HW NED": round(report.get('avg_handwriting_ned', 0.0), 4),
                        "Samples": report.get('sample_count', 0)
                    })
                else:
                    results.append({
                        "Model ID": model_id,
                        "Avg CER": round(report.get('average_cer', 0.0), 4),
                        "Avg WER": round(report.get('average_wer', 0.0), 4),
                        "Avg NED": round(report.get('average_ned', 0.0), 4),
                        "Precision": round(report.get('average_precision', 0.0), 4),
                        "Recall": round(report.get('average_recall', 0.0), 4),
                        "BoW F1": round(report.get('average_bow_f1', 0.0), 4),
                        "Exact Match": round(report.get('exact_match_accuracy', 0.0), 4),
                        "Samples": report.get('sample_count', 0)
                    })
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
                
                if v_key == "v2":
                    results.append({
                        "Model ID": model_id,
                        "Weighted Score": round(report['avg_weighted_score'], 4),
                        "Y/N Acc": round(report['avg_yn_acc'], 4),
                        "HW CER": round(report['avg_handwriting_cer'], 4),
                        "HW WER": round(report['avg_handwriting_wer'], 4),
                        "HW NED": round(report['avg_handwriting_ned'], 4),
                        "Samples": report['sample_count']
                    })
                else:
                    results.append({
                        "Model ID": model_id,
                        "Avg CER": round(report['average_cer'], 4),
                        "Avg WER": round(report['average_wer'], 4),
                        "Avg NED": round(report['average_ned'], 4),
                        "Precision": round(report['average_precision'], 4),
                        "Recall": round(report['average_recall'], 4),
                        "BoW F1": round(report['average_bow_f1'], 4),
                        "Exact Match": round(report['exact_match_accuracy'], 4),
                        "Samples": report['sample_count']
                    })
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
    else:
        st.info(f"No {v_key.upper()} results found. Run a benchmark to see data.")

# --- Tab 2: Detailed View ---
with tab2:
    if results_data:
        st.header("🔍 Detailed Comparison")
        
        selected_image = st.selectbox("Select Image to Inspect", list(gt_dict.keys()))
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
                res_file = f"results/preds_{v_key}_{mid}.json"
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
                    res_file = f"results/preds_{v_key}_{mid}.json"
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
    else:
        st.info(f"No {v_key.upper()} results found. Run a benchmark to see data.")

# --- Tab 3: Statistical Analysis ---
with tab3:
    if results_data and len(results_data) >= 2:
        st.header("📈 Statistical Analysis")
        
        full_results = load_full_results(v_key)
        
        if len(full_results) >= 2:
            # Select metric for comparison
            if v_key == "v2":
                metric_options = ["weighted_score", "yn_acc", "handwriting_cer", "handwriting_wer", "handwriting_ned"]
                default_metric = "weighted_score"
            else:
                metric_options = ["cer", "wer", "ned", "bow_f1", "exact_match"]
                default_metric = "cer"
            
            selected_metric = st.selectbox("Select Metric for Analysis", metric_options, index=0)
            
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
                    with col1:
                        st.metric(
                            f"{model1}",
                            f"{comparison['model1']['mean']:.4f}",
                            help=f"95% CI: [{comparison['model1']['ci_95'][0]:.4f}, {comparison['model1']['ci_95'][1]:.4f}]"
                        )
                    with col2:
                        st.metric(
                            f"{model2}",
                            f"{comparison['model2']['mean']:.4f}",
                            help=f"95% CI: [{comparison['model2']['ci_95'][0]:.4f}, {comparison['model2']['ci_95'][1]:.4f}]"
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
                
                comparison_data = []
                for comp in comparisons:
                    comparison_data.append({
                        'Model 1': comp['model1_id'],
                        'Model 2': comp['model2_id'],
                        'M1 Mean': f"{comp['model1']['mean']:.4f}",
                        'M2 Mean': f"{comp['model2']['mean']:.4f}",
                        'p-value': f"{comp['statistical_test'].get('p_value', 0):.6f}",
                        'Significant': '✓' if comp['statistical_test'].get('significant', False) else '✗',
                        'Winner': comp.get('winner', 'N/A')
                    })
                
                comp_df = pd.DataFrame(comparison_data)
                st.dataframe(comp_df, width='stretch')
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
