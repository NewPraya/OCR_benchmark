import streamlit as st

from main import run_benchmark

from dashboard.data_loader import load_all_results
from dashboard.tabs import detailed_tab, export_tab, leaderboard_tab, multirun_tab, stats_tab

st.set_page_config(page_title="OCR Benchmark Dashboard", layout="wide")
st.title("📊 OCR Benchmark Dashboard")
st.caption(
    "Explore single-run reports from `results/` and optional multi-run summaries from "
    "`results/multirun/`."
)

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
runs_per_image = st.sidebar.number_input("Runs Per Image", min_value=1, max_value=20, value=1, step=1)
st.sidebar.caption(
    "Use `1` for the standard benchmark path. Values greater than `1` enable multi-run output files "
    "under `results/multirun/`."
)
run_btn = st.sidebar.button("🚀 Run Benchmark")

if run_btn:
    model_ids = [m.strip() for m in model_ids_input.split("\n") if m.strip()]
    with st.spinner(f"Running {v_key.upper()} benchmark for {len(model_ids)} models..."):
        run_benchmark(
            model_type,
            model_ids,
            eval_version=v_key,
            postprocess=(not disable_postprocess),
            runs_per_image=int(runs_per_image),
        )
    st.success("Benchmark completed!")

results_data, gt_dict = load_all_results(v_key)

if not results_data:
    st.info(
        "No benchmark reports were found for this evaluation mode yet. Run a model from the sidebar, "
        "or place `results/preds_*` / `results/report_*` files in the repository root."
    )

# --- Tabs for better organization ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Leaderboard", "🔍 Detailed View", "📈 Statistical Analysis", "📤 Export", "🔁 Multi-run"]
)

with tab1:
    leaderboard_tab.render(v_key, results_data)

with tab2:
    detailed_tab.render(v_key, results_data, gt_dict)

with tab3:
    stats_tab.render(v_key, results_data)

with tab4:
    export_tab.render(v_key, results_data)

with tab5:
    multirun_tab.render()
