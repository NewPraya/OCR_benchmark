import io
import json
import os

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from dashboard.utils import (
    base_model_id,
    file_signature,
    model_family_from_id,
)


sns.set_style("whitegrid")
EXCLUDED_DISPLAY_MODELS = {"gemini-3-pro"}

FAMILY_PALETTE = {
    "Gemini": "#1f77b4",
    "OpenAI": "#ff7f0e",
    "Qwen": "#2ca02c",
    "Claude": "#d62728",
    "Ollama/Llama": "#9467bd",
    "Other": "#7f7f7f",
}


def _to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def _display_model_name(model_id: str) -> str:
    name = base_model_id(model_id)
    name = name.replace("-preview", "")
    return name


def _is_excluded_model(base_model_id_val: str) -> bool:
    return _display_model_name(base_model_id_val) in EXCLUDED_DISPLAY_MODELS


@st.cache_data(show_spinner=False)
def _load_multirun_summary_cached(v_key: str, postprocess_enabled: bool, leaderboard_sig):
    leaderboard_path = f"results/multirun/leaderboard_{v_key}.json"
    metric_col = "Avg CER" if v_key == "v1" else "Weighted Score"

    if not os.path.exists(leaderboard_path):
        return pd.DataFrame()

    try:
        with open(leaderboard_path, "r") as f:
            rows = json.load(f)
    except Exception:
        return pd.DataFrame()

    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        model_id = row.get("Model ID")
        if not model_id:
            continue
        base_id = base_model_id(model_id)
        if str(base_id).startswith("dummy") or _is_excluded_model(base_id):
            continue
        val = row.get(metric_col)
        if val is None:
            continue
        if bool(row.get("Postprocess", True)) != bool(postprocess_enabled):
            continue
        out.append(
            {
                "Model ID": model_id,
                "Base Model": base_id,
                "Display Model": _display_model_name(model_id),
                "Family": model_family_from_id(model_id),
                "Metric": float(val),
                "Postprocess": bool(row.get("Postprocess", True)),
            }
        )

    return pd.DataFrame(out)


def load_multirun_summary(v_key: str, postprocess_enabled: bool) -> pd.DataFrame:
    return _load_multirun_summary_cached(
        v_key,
        postprocess_enabled,
        file_signature(f"results/multirun/leaderboard_{v_key}.json"),
    )


@st.cache_data(show_spinner=False)
def _load_task_distribution_multirun_cached(v_key: str, dist_sig):
    dist_path = f"results/multirun/distribution_{v_key}.json"
    if not os.path.exists(dist_path):
        return pd.DataFrame()

    try:
        with open(dist_path, "r") as f:
            rows = json.load(f)
    except Exception:
        return pd.DataFrame()

    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        display_model = str(row.get("Display Model") or "")
        base_id = str(row.get("Base Model") or "")
        if not display_model or not base_id:
            continue
        if str(base_id).startswith("dummy") or _is_excluded_model(base_id):
            continue
        score = row.get("Score")
        run_idx = row.get("Run")
        if score is None or run_idx is None:
            continue
        try:
            out.append(
                {
                    "Base Model": base_id,
                    "Display Model": display_model,
                    "Family": str(row.get("Family") or model_family_from_id(base_id)),
                    "Score": float(score),
                    "Run": int(run_idx),
                }
            )
        except Exception:
            continue

    return pd.DataFrame(out)


def load_task_distribution_multirun(v_key: str) -> pd.DataFrame:
    dist_sig = file_signature(f"results/multirun/distribution_{v_key}.json")
    return _load_task_distribution_multirun_cached(v_key, dist_sig)


def _build_combined_ablation_df(v1_on: pd.DataFrame, v2_on: pd.DataFrame, v1_off: pd.DataFrame, v2_off: pd.DataFrame):
    if v1_on.empty or v2_on.empty:
        return pd.DataFrame()

    on_v1 = v1_on.groupby("Base Model", as_index=False).agg({"Metric": "mean", "Display Model": "first", "Family": "first"})
    on_v1 = on_v1.rename(columns={"Metric": "CER ON"})
    on_v2 = v2_on.groupby("Base Model", as_index=False).agg({"Metric": "mean"}).rename(columns={"Metric": "Weighted ON"})

    merged = on_v1.merge(on_v2, on="Base Model", how="inner")
    if merged.empty:
        return pd.DataFrame()

    if not v1_off.empty:
        off_v1 = v1_off.rename(columns={"Metric": "CER OFF"})
        merged = merged.merge(off_v1, on="Base Model", how="left")
    else:
        merged["CER OFF"] = float("nan")

    if not v2_off.empty:
        off_v2 = v2_off.rename(columns={"Metric": "Weighted OFF"})
        merged = merged.merge(off_v2, on="Base Model", how="left")
    else:
        merged["Weighted OFF"] = float("nan")

    return merged.sort_values("CER ON", ascending=True).reset_index(drop=True)


def _plot_combined_ablation(ax, merged_df: pd.DataFrame):
    if merged_df.empty:
        ax.text(0.5, 0.5, "No multi-run ON summary found", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    x = list(range(len(merged_df)))
    width = 0.34

    handles = []
    labels = []

    off_vals = []
    if merged_df["CER OFF"].notna().any():
        off_vals = merged_df["CER OFF"].tolist()
        h = ax.bar(
            [i - width / 2 for i in x],
            off_vals,
            width=width,
            label="Transcription Task (Avg CER) - OFF",
            color="#6baed6",
        )
        handles.append(h)
        labels.append("Transcription Task (Avg CER) - OFF")
        on_pos = [i + width / 2 for i in x]
    else:
        on_pos = x

    on_vals = merged_df["CER ON"].tolist()
    h_on = ax.bar(
        on_pos,
        on_vals,
        width=width,
        label="Transcription Task (Avg CER) - ON",
        color="#2171b5",
    )
    handles.append(h_on)
    labels.append("Transcription Task (Avg CER) - ON")

    ax2 = ax.twinx()
    if merged_df["Weighted OFF"].notna().any():
        line_off = ax2.plot(
            x,
            merged_df["Weighted OFF"].tolist(),
            marker="o",
            linewidth=1.8,
            color="#fdae6b",
            label="Structured Extraction Task (Weighted Score) - OFF",
        )
        handles.append(line_off[0])
        labels.append("Structured Extraction Task (Weighted Score) - OFF")

    line_on = ax2.plot(
        x,
        merged_df["Weighted ON"].tolist(),
        marker="o",
        linewidth=2.2,
        color="#e6550d",
        label="Structured Extraction Task (Weighted Score) - ON",
    )
    handles.append(line_on[0])
    labels.append("Structured Extraction Task (Weighted Score) - ON")

    ax.set_xticks(x)
    ax.set_xticklabels(merged_df["Display Model"].tolist(), rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Transcription Task: Avg CER")
    ax2.set_ylabel("Structured Extraction Task: Weighted Score")
    ax.grid(True, axis="y", alpha=0.25)

    # Focus CER visual range on [0, 1.0] for readability; annotate truncated bars above the cap.
    cer_cap = 1.0
    max_cer = max((off_vals + on_vals), default=0.0)
    if max_cer > cer_cap:
        ax.set_ylim(0.0, cer_cap)
        for xi, val in zip(on_pos, on_vals):
            if val > cer_cap:
                ax.text(
                    xi,
                    cer_cap + 0.01,
                    f"↑{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#2171b5",
                    clip_on=False,
                )
        if off_vals:
            for xi, val in zip([i - width / 2 for i in x], off_vals):
                if val > cer_cap:
                    ax.text(
                        xi,
                        cer_cap + 0.01,
                        f"↑{val:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color="#6baed6",
                        clip_on=False,
                    )

    legend_pairs = []
    for idx in [2, 3, 0, 1]:
        if idx < len(handles) and idx < len(labels):
            legend_pairs.append((handles[idx], labels[idx]))
    legend_handles = [h for h, _ in legend_pairs]
    legend_labels = [l for _, l in legend_pairs]

    legend = ax2.legend(
        legend_handles,
        legend_labels,
        fontsize=8,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.98),
        ncol=1,
        frameon=True,
        facecolor="none",
        framealpha=1.0,
    )
    legend.get_frame().set_facecolor("none")
    legend.set_zorder(30)


def _build_order_df(v1_on: pd.DataFrame, v2_on: pd.DataFrame):
    if v1_on.empty or v2_on.empty:
        return pd.DataFrame()
    left = v1_on.groupby("Base Model", as_index=False).agg({"Metric": "mean", "Display Model": "first", "Family": "first"})
    right = v2_on.groupby("Base Model", as_index=False).agg({"Metric": "mean"})
    merged = left.merge(right, on="Base Model", how="inner")
    if merged.empty:
        return pd.DataFrame()
    return merged.sort_values("Metric_x", ascending=True)[["Base Model", "Display Model", "Family"]]


def _plot_dual_distribution(ax_left, ax_right, v1_dist_df: pd.DataFrame, v2_dist_df: pd.DataFrame, order_df: pd.DataFrame):
    if order_df.empty:
        ax_left.text(0.5, 0.5, "No shared ON data", ha="center", va="center", transform=ax_left.transAxes)
        ax_left.set_axis_off()
        ax_right.set_axis_off()
        return

    order = order_df["Display Model"].tolist()
    family_map = {row["Display Model"]: row["Family"] for _, row in order_df.iterrows()}
    palette = {m: FAMILY_PALETTE.get(family_map.get(m, "Other"), FAMILY_PALETTE["Other"]) for m in order}

    left_plot_df = v1_dist_df[v1_dist_df["Display Model"].isin(order)].copy()
    right_plot_df = v2_dist_df[v2_dist_df["Display Model"].isin(order)].copy()

    sns.boxplot(
        data=left_plot_df,
        x="Score",
        y="Display Model",
        order=order,
        orient="h",
        palette=palette,
        linewidth=0.8,
        fliersize=1.2,
        ax=ax_left,
    )
    ax_left.set_xlabel("Transcription Task: CER (lower is better)")
    ax_left.set_ylabel("Model")
    ax_left.tick_params(axis="y", labelsize=8)
    ax_left.grid(True, axis="x", alpha=0.25)
    if not left_plot_df.empty:
        q99 = float(left_plot_df["Score"].quantile(0.99))
        ax_left.set_xlim(0.0, max(0.2, q99 * 1.15))

    sns.boxplot(
        data=right_plot_df,
        x="Score",
        y="Display Model",
        order=order,
        orient="h",
        palette=palette,
        linewidth=0.8,
        fliersize=1.2,
        ax=ax_right,
    )
    ax_right.set_xlabel("Structured Extraction Task: Weighted Score (higher is better)")
    ax_right.grid(True, axis="x", alpha=0.25)
    ax_right.tick_params(axis="y", labelleft=False)
    ax_right.set_xlim(0.0, 1.0)

    families = sorted(set(order_df["Family"].tolist()))
    handles = [
        mlines.Line2D([], [], color=FAMILY_PALETTE.get(f, FAMILY_PALETTE["Other"]), marker="o", linestyle="None", label=f)
        for f in families
    ]
    legend = ax_right.legend(handles=handles, title="Model Family", fontsize=8, title_fontsize=8, loc="upper left")
    ax_right.add_artist(legend)


def render_paper_figures():
    v1_on = load_multirun_summary("v1", postprocess_enabled=True)
    v2_on = load_multirun_summary("v2", postprocess_enabled=True)
    v1_off = load_multirun_summary("v1", postprocess_enabled=False)
    v2_off = load_multirun_summary("v2", postprocess_enabled=False)
    v1_dist = load_task_distribution_multirun("v1")
    v2_dist = load_task_distribution_multirun("v2")

    st.subheader("Paper Figures")
    st.caption(
        "Optional reproduction helpers built on top of precomputed multi-run summaries. "
        "These views are not part of the benchmark's required execution path."
    )

    title_y = 0.94

    fig1, ax1 = plt.subplots(1, 1, figsize=(10.2, 8.6))
    fig1.suptitle("Combined Ablation: Transcription Task + Structured Extraction Task", fontsize=11, y=title_y)
    _plot_combined_ablation(ax1, _build_combined_ablation_df(v1_on, v2_on, v1_off, v2_off))
    fig1.subplots_adjust(left=0.085, right=0.985, bottom=0.08, top=0.92)
    st.pyplot(fig1)
    st.download_button(
        "Download Combined Ablation Figure (PNG)",
        data=_to_png_bytes(fig1),
        file_name="combined_ablation_transcription_structured.png",
        mime="image/png",
        key="download_fig_combined_ablation",
    )
    plt.close(fig1)

    # Keep figure-2 canvas consistent with figure-1 for side-by-side paper layout.
    fig2, (ax2_left, ax2_right) = plt.subplots(1, 2, figsize=(10.2, 8.6), sharey=True, gridspec_kw={"wspace": 0.04})
    fig2.suptitle("Per-sample Score Distributions by Model (Postprocess ON)", fontsize=11, y=title_y)
    _plot_dual_distribution(ax2_left, ax2_right, v1_dist, v2_dist, _build_order_df(v1_on, v2_on))
    # Use explicit subplot bounds so title-to-axes spacing is stable for export layout.
    fig2.subplots_adjust(left=0.085, right=0.985, bottom=0.08, top=0.92, wspace=0.04)
    st.pyplot(fig2)
    st.download_button(
        "Download Dual-metric Row Figure (PNG)",
        data=_to_png_bytes(fig2),
        file_name="dual_metric_row_cer_weighted.png",
        mime="image/png",
        key="download_fig_dual_metric_row",
    )
    plt.close(fig2)
