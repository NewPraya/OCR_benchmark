import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from dashboard.utils import base_model_id, model_family_from_id


def render(v_key: str, results_data):
    if not results_data:
        st.info(f"No {v_key.upper()} results found. Run a benchmark to see data.")
        return None

    st.header(f"📈 Leaderboard ({v_key.upper()})")
    df = pd.DataFrame(results_data)
    sort_col = "Weighted Score" if v_key == "v2" else "Avg CER"
    ascending = v_key != "v2"
    df = df.sort_values(sort_col, ascending=ascending)
    st.dataframe(df, width="stretch")

    st.subheader("Summary Statistics")
    st.write(df.describe())

    if "Postprocess" in df.columns:
        chart_df = df.copy()
        chart_df["Base Model"] = chart_df["Model ID"].apply(base_model_id)
        paired = chart_df.groupby("Base Model")["Postprocess"].nunique()
        paired_models = paired[paired >= 2].index.tolist()

        if paired_models:
            st.subheader("Postprocess Ablation (Same Model Paired)")

            metric_candidates = [
                c
                for c in chart_df.columns
                if c
                not in {
                    "Model ID",
                    "Base Model",
                    "Postprocess",
                    "Samples",
                    "Target",
                    "Processed",
                    "Failed",
                    "Failed Rate",
                }
                and pd.api.types.is_numeric_dtype(chart_df[c])
            ]
            default_metric = "Weighted Score" if v_key == "v2" else "Avg CER"
            metric_idx = metric_candidates.index(default_metric) if default_metric in metric_candidates else 0
            selected_metric = st.selectbox(
                "Metric for paired bar chart",
                metric_candidates,
                index=metric_idx,
                key=f"paired_bar_metric_{v_key}",
            )

            lower_is_better = selected_metric in {
                "Avg CER",
                "Avg WER",
                "Avg NED",
                "HW CER",
                "HW WER",
                "HW NED",
            }

            paired_df = chart_df[chart_df["Base Model"].isin(paired_models)]
            pivot = paired_df.pivot_table(index="Base Model", columns="Postprocess", values=selected_metric, aggfunc="mean")

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
            ax.set_xticklabels(pivot.index.tolist(), rotation=35, ha="right")
            ax.set_ylabel(selected_metric)
            ax.set_title(f"Paired Ablation by Model ({selected_metric})")
            ax.grid(True, axis="y", alpha=0.3)
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

            delta_rows = []
            for model_name, row in pivot.iterrows():
                on_val = row.get(True, None)
                off_val = row.get(False, None)
                if on_val is None or off_val is None:
                    continue
                delta_rows.append(
                    {
                        "Base Model": model_name,
                        "Postprocess ON": round(float(on_val), 4),
                        "Postprocess OFF": round(float(off_val), 4),
                        "Delta (ON - OFF)": round(float(on_val - off_val), 4),
                    }
                )
            if delta_rows:
                st.caption(
                    "Delta > 0 means ON is numerically higher; for error metrics (CER/WER/NED), lower is better."
                )
                st.dataframe(pd.DataFrame(delta_rows), width="stretch")

    st.divider()
    st.subheader("Custom Chart Builder")
    st.caption("Filter model scope and metrics, then plot freely for paper figures.")

    custom_df = df.copy()
    custom_df["Base Model"] = custom_df["Model ID"].apply(base_model_id)
    custom_df["Model Family"] = custom_df["Model ID"].apply(model_family_from_id)
    custom_df["Postprocess Label"] = custom_df["Postprocess"].apply(lambda x: "ON" if bool(x) else "OFF")

    family_options = sorted(custom_df["Model Family"].unique().tolist())
    selected_families = st.multiselect("Model families", options=family_options, default=family_options)

    state_options = ["ON", "OFF"] if "Postprocess" in custom_df.columns else ["ON"]
    selected_states = st.multiselect("Postprocess states", options=state_options, default=state_options)

    family_filtered_df = custom_df[
        custom_df["Model Family"].isin(selected_families) & custom_df["Postprocess Label"].isin(selected_states)
    ]

    model_options = family_filtered_df["Model ID"].tolist()
    selected_models = st.multiselect("Specific models (optional, choose subset)", options=model_options, default=model_options)
    plot_df = family_filtered_df[family_filtered_df["Model ID"].isin(selected_models)]

    metric_options = [
        c
        for c in plot_df.columns
        if c
        not in {
            "Model ID",
            "Base Model",
            "Model Family",
            "Postprocess",
            "Postprocess Label",
            "Samples",
            "Target",
            "Processed",
            "Failed",
            "Failed Rate",
        }
        and pd.api.types.is_numeric_dtype(plot_df[c])
    ]
    default_metric = "Weighted Score" if v_key == "v2" else "Avg CER"
    selected_metrics = st.multiselect(
        "Metrics",
        options=metric_options,
        default=[default_metric] if default_metric in metric_options else metric_options[:1],
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
            squeeze=False,
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
                    ax=ax,
                )
            else:
                sns.lineplot(
                    data=metric_plot_df,
                    x="Model ID",
                    y=metric,
                    hue="Postprocess Label",
                    marker="o",
                    ax=ax,
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
        st.dataframe(export_df, width="stretch")
        st.download_button(
            label="Download filtered chart data (CSV)",
            data=export_df.to_csv(index=False),
            file_name=f"custom_plot_data_{v_key}.csv",
            mime="text/csv",
        )

    return df
