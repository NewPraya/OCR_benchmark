import pandas as pd
import streamlit as st

from dashboard.data_loader import load_full_results
from dashboard.utils import build_export_bundle, export_to_latex


def render(v_key: str, results_data):
    if not results_data:
        st.info("No results available for export.")
        return

    st.header("📤 Export Results")

    st.subheader("LaTeX Table")
    df = pd.DataFrame(results_data)

    caption = st.text_input("Table Caption", f"OCR Benchmark Results ({v_key.upper()} Mode)")

    if st.button("Generate LaTeX"):
        latex_code = export_to_latex(df, caption=caption)
        st.code(latex_code, language="latex")
        st.download_button(
            label="Download LaTeX",
            data=latex_code,
            file_name=f"ocr_results_{v_key}.tex",
            mime="text/plain",
        )

    st.subheader("CSV Export")
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"ocr_results_{v_key}.csv",
        mime="text/csv",
    )

    st.subheader("JSON Export")
    json_str = df.to_json(orient="records", indent=2)
    st.download_button(
        label="Download JSON",
        data=json_str,
        file_name=f"ocr_results_{v_key}.json",
        mime="application/json",
    )

    st.divider()
    st.subheader("One-click Paper Bundle")
    st.caption("Package leaderboard exports + raw reports + pairwise stats into one zip file.")
    bundle_metric = "weighted_score" if v_key == "v2" else "cer"
    bundle_caption = st.text_input(
        "Bundle table caption",
        f"OCR Benchmark Results ({v_key.upper()} Mode)",
        key=f"bundle_caption_{v_key}",
    )
    if st.button("Prepare ZIP Bundle", key=f"prepare_zip_{v_key}"):
        with st.spinner("Building zip bundle..."):
            full_results = load_full_results(v_key)
            bundle_bytes = build_export_bundle(
                v_key=v_key,
                df=df,
                caption=bundle_caption,
                full_results=full_results,
                stats_metric=bundle_metric,
            )
            st.session_state[f"bundle_bytes_{v_key}"] = bundle_bytes
        st.success("Bundle prepared. Click download below.")

    if st.session_state.get(f"bundle_bytes_{v_key}"):
        st.download_button(
            label="Download Paper Bundle (.zip)",
            data=st.session_state[f"bundle_bytes_{v_key}"],
            file_name=f"meddocbench_{v_key}_paper_bundle.zip",
            mime="application/zip",
            key=f"download_zip_{v_key}",
        )
