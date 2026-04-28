import streamlit as st

from dashboard.multirun_loader import load_multirun_precomputed
from dashboard.utils import has_optional_reproduction_assets


def _render_version(v_key: str):
    leaderboard_df, per_run_df, std_df, meta = load_multirun_precomputed(v_key)

    if leaderboard_df.empty:
        st.warning(
            f"{v_key.upper()} summary not found. Please precompute first:\n"
            "`python utils/generate_multirun_summary.py --version all --write-csv`"
        )
        return

    st.subheader(f"Leaderboard ({v_key.upper()})")
    st.dataframe(leaderboard_df, width="stretch")

    if meta:
        gen_time = meta.get("generated_at_utc", "N/A")
        source_files = meta.get("source_files", "N/A")
        st.caption(f"Generated at (UTC): {gen_time} | Source files: {source_files}")

    st.download_button(
        label=f"Download {v_key.upper()} leaderboard (CSV)",
        data=leaderboard_df.to_csv(index=False),
        file_name=f"multirun_leaderboard_{v_key}.csv",
        mime="text/csv",
        key=f"download_multirun_leaderboard_{v_key}",
    )

    with st.expander(f"Per-run table ({v_key.upper()})", expanded=False):
        st.dataframe(per_run_df, width="stretch")
        st.download_button(
            label=f"Download {v_key.upper()} per-run table (CSV)",
            data=per_run_df.to_csv(index=False),
            file_name=f"multirun_per_run_{v_key}.csv",
            mime="text/csv",
            key=f"download_multirun_perrun_{v_key}",
        )

    if not std_df.empty:
        with st.expander(f"Std table ({v_key.upper()})", expanded=False):
            st.dataframe(std_df, width="stretch")


def render():
    st.header("🔁 Multi-run")
    st.caption(
        "This tab reads precomputed summary artifacts from `results/multirun/`. "
        "Generate them with `python utils/generate_multirun_summary.py --version all --write-csv`."
    )

    tab_v1, tab_v2 = st.tabs(["V1 Leaderboard", "V2 Leaderboard"])

    with tab_v1:
        _render_version("v1")

    with tab_v2:
        _render_version("v2")

    if has_optional_reproduction_assets():
        st.divider()
        with st.expander("Optional reproduction figures", expanded=False):
            st.caption(
                "These figure views are convenience tools for paper reproduction. "
                "They are not required for the benchmark or dashboard core workflow."
            )
            from dashboard.paper_figures import render_paper_figures

            render_paper_figures()
