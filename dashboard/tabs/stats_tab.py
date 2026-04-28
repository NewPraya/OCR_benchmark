import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from evaluators.statistical_tests import compare_models, batch_compare_models

from dashboard.data_loader import load_full_results
from dashboard.utils import build_pairwise_table, format_ci, stats_metric_options


def render(v_key: str, results_data):
    if not results_data or len(results_data) < 2:
        st.info("Need at least 2 models with results for statistical analysis.")
        return

    st.header("📈 Statistical Analysis")
    full_results = load_full_results(v_key)

    if len(full_results) < 2:
        st.info("Need at least 2 models with results for statistical comparison.")
        return

    metric_options, default_metric = stats_metric_options(v_key)
    metric_idx = metric_options.index(default_metric) if default_metric in metric_options else 0
    selected_metric = st.selectbox("Select Metric for Analysis", metric_options, index=metric_idx)

    model_ids = list(full_results.keys())
    col1, col2 = st.columns(2)
    with col1:
        model1 = st.selectbox("Model 1", model_ids, index=0)
    with col2:
        model2 = st.selectbox("Model 2", model_ids, index=min(1, len(model_ids) - 1))

    use_parametric = st.checkbox(
        "Use parametric test (t-test)", value=True, help="Uncheck to use non-parametric Wilcoxon test"
    )

    if st.button("🔬 Run Statistical Comparison"):
        if model1 == model2:
            st.warning("Please select two different models to compare.")
        else:
            with st.spinner("Running statistical tests..."):
                comparison = compare_models(
                    full_results[model1],
                    full_results[model2],
                    metric_name=selected_metric,
                    use_parametric=use_parametric,
                )

            st.subheader("Comparison Results")

            col1, col2, col3 = st.columns(3)
            m1_ci = comparison.get("model1", {}).get("ci_95", (None, None))
            m2_ci = comparison.get("model2", {}).get("ci_95", (None, None))
            with col1:
                st.metric(
                    f"{model1}",
                    f"{comparison['model1']['mean']:.4f}",
                    help=f"95% CI: {format_ci(m1_ci)}",
                )
            with col2:
                st.metric(
                    f"{model2}",
                    f"{comparison['model2']['mean']:.4f}",
                    help=f"95% CI: {format_ci(m2_ci)}",
                )
            with col3:
                st.metric("Winner", comparison.get("winner", "N/A"))

            st.subheader("Statistical Test")
            test_result = comparison["statistical_test"]

            test_df = pd.DataFrame(
                [
                    {
                        "Test": test_result.get("test", "N/A"),
                        "p-value": f"{test_result.get('p_value', 0):.6f}",
                        "Significant": "✓" if test_result.get("significant", False) else "✗",
                        "Interpretation": test_result.get("interpretation", "N/A"),
                    }
                ]
            )
            st.dataframe(test_df, width="stretch")

            st.subheader("Score Distribution")
            fig, ax = plt.subplots(figsize=(10, 5))
            data_to_plot = [comparison["model1"]["scores"], comparison["model2"]["scores"]]
            ax.boxplot(data_to_plot, labels=[model1, model2])
            ax.set_ylabel(selected_metric)
            ax.set_title(f"{selected_metric} Distribution Comparison")
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)
            plt.close(fig)

    st.divider()
    st.subheader("Pairwise Comparisons (All Models)")

    if st.button("🔬 Run All Pairwise Comparisons"):
        with st.spinner("Running pairwise comparisons..."):
            comparisons = batch_compare_models(full_results, selected_metric, use_parametric)

        comp_df = build_pairwise_table(comparisons)
        st.dataframe(comp_df, width="stretch")
        st.download_button(
            label="Download pairwise stats (CSV)",
            data=comp_df.to_csv(index=False),
            file_name=f"pairwise_stats_{v_key}_{selected_metric}.csv",
            mime="text/csv",
            key=f"download_pairwise_{v_key}_{selected_metric}",
        )
