import streamlit as st
import json
import os
import pandas as pd
from PIL import Image
import glob
from main import run_benchmark
from evaluators.evaluator import OCREvaluator
from evaluators.evaluator_v2 import OCREvaluatorV2

st.set_page_config(page_title="OCR Benchmark Dashboard", layout="wide")

st.title("📊 OCR Benchmark Dashboard")

# --- Sidebar: Configuration ---
st.sidebar.header("Run Benchmark")
eval_version = st.sidebar.radio("Evaluation Mode", ["v1 (Text)", "v2 (Structured)"], index=0)
v_key = "v1" if "v1" in eval_version else "v2"

model_type = st.sidebar.selectbox("Select Model Type", ["gemini", "qwen", "openai", "dummy"])
default_ids = "gemini-2.0-flash-exp" if model_type == "gemini" else "gpt-4o"
model_ids_input = st.sidebar.text_area("Model IDs (one per line)", value=default_ids)
run_btn = st.sidebar.button("🚀 Run Benchmark")

if run_btn:
    model_ids = [m.strip() for m in model_ids_input.split("\n") if m.strip()]
    with st.spinner(f"Running {v_key.upper()} benchmark for {len(model_ids)} models..."):
        run_benchmark(model_type, model_ids, eval_version=v_key)
    st.success("Benchmark completed!")

# --- Load Results ---
def load_all_results(v_key):
    results = []
    # Search for files matching the version
    result_files = glob.glob(f"results/preds_{v_key}_*.json")
    
    gt_path = "data/sample_gt_v2.json" if v_key == "v2" else "data/sample_gt.json"
    if not os.path.exists(gt_path):
        return [], {}

    if v_key == "v2":
        evaluator = OCREvaluatorV2(gt_path)
    else:
        evaluator = OCREvaluator(gt_path)
    
    for f in result_files:
        model_id = os.path.basename(f).replace(f"preds_{v_key}_", "").replace(".json", "")
        with open(f, 'r') as j:
            try:
                predictions = json.load(j)
                if not predictions: continue
                report = evaluator.evaluate_results(predictions)
                
                if v_key == "v2":
                    results.append({
                        "Model ID": model_id,
                        "Logical Acc": round(report['avg_logical_acc'], 4),
                        "Entity Recall": round(report['avg_entity_recall'], 4),
                        "Pairing Acc": round(report['avg_pairing_acc'], 4),
                        "Samples": report['sample_count']
                    })
                else:
                    results.append({
                        "Model ID": model_id,
                        "Avg CER": round(report['average_cer'], 4),
                        "Avg WER": round(report['average_wer'], 4),
                        "Samples": report['sample_count']
                    })
            except Exception:
                continue
    return results, evaluator.gt_dict

results_data, gt_dict = load_all_results(v_key)

# --- Main UI: Overview ---
if results_data:
    st.header(f"📈 Leaderboard ({v_key.upper()})")
    df = pd.DataFrame(results_data)
    sort_col = "Logical Acc" if v_key == "v2" else "Avg CER"
    ascending = False if v_key == "v2" else True
    df = df.sort_values(sort_col, ascending=ascending)
    st.dataframe(df, width="stretch")

    # --- Detail View: Compare Models ---
    st.divider()
    st.header("🔍 Detailed Comparison")
    
    selected_image = st.selectbox("Select Image to Inspect", list(gt_dict.keys()))
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Original Image")
        img_path = os.path.join("data", selected_image)
        if os.path.exists(img_path):
            st.image(img_path, width="stretch")
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
                with open(res_file, 'r') as f:
                    preds = json.load(f)
                    pred_item = next((p for p in preds if p['file_name'] == selected_image), None)
                    if pred_item:
                        with st.expander(f"Model: {mid}", expanded=True):
                            if v_key == "v2":
                                try:
                                    pred_json = json.loads(pred_item['prediction']) if isinstance(pred_item['prediction'], str) else pred_item['prediction']
                                    st.json(pred_json)
                                except:
                                    st.error("Failed to parse JSON")
                                    st.text(pred_item['prediction'])
                            else:
                                st.text_area(f"Prediction ({mid})", pred_item['prediction'], height=200)
            else:
                st.warning(f"No results for model {mid}")
else:
    st.info(f"No {v_key.upper()} results found. Run a benchmark to see data.")

# --- Tools ---
st.sidebar.divider()
st.sidebar.header("Tools")
if st.sidebar.button("🔄 Sync MD to GT (Structured V2)"):
    from utils.sync_to_gt import sync_to_gt
    sync_to_gt()
    st.sidebar.success("Synced labels to V2 GT!")
    st.rerun()
