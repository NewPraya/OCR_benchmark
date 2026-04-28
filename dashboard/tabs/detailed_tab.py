import json
import os

import pandas as pd
import streamlit as st

from evaluators.evaluator_v2 import OCREvaluatorV2

from dashboard.data_loader import load_full_results, load_predictions_file_cached
from dashboard.utils import build_case_finder_df, file_signature, resolve_gt_path, resolve_pred_path


def render(v_key: str, results_data, gt_dict):
    if not results_data:
        st.info(f"No {v_key.upper()} results found. Run a benchmark to see data.")
        return

    st.header("🔍 Detailed Comparison")

    detail_key = f"detail_selected_image_{v_key}"
    image_options = list(gt_dict.keys())
    if not image_options:
        st.info("No images available in the current split.")
        return
    if image_options:
        if detail_key not in st.session_state or st.session_state[detail_key] not in image_options:
            st.session_state[detail_key] = image_options[0]
    selected_image = st.selectbox("Select Image to Inspect", image_options, key=detail_key)
    gt_path = resolve_gt_path(v_key)

    df = pd.DataFrame(results_data)

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
        selected_models = st.multiselect(
            "Select Models to Compare",
            df["Model ID"].tolist(),
            default=df["Model ID"].tolist()[:2],
        )

        for mid in selected_models:
            res_file = resolve_pred_path(v_key, mid)
            if os.path.exists(res_file):
                preds = load_predictions_file_cached(res_file, file_signature(res_file))
                pred_item = next((p for p in preds if p["file_name"] == selected_image), None)
                if pred_item:
                    with st.expander(f"Model: {mid}", expanded=True):
                        if v_key == "v2":
                            try:
                                pred_json = (
                                    json.loads(pred_item["prediction"])
                                    if isinstance(pred_item["prediction"], str)
                                    else pred_item["prediction"]
                                )
                                st.json(pred_json)
                            except Exception:
                                st.error("Failed to parse JSON")
                                st.text(pred_item["prediction"])
                        else:
                            st.text_area(f"Prediction ({mid})", pred_item["prediction"], height=200, key=f"pred_{mid}")
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
                "GT": evaluator._normalize_yn(gt_value) or str(gt_value),
            }
            for mid in selected_models:
                res_file = resolve_pred_path(v_key, mid)
                pred_value = "-"
                if os.path.exists(res_file):
                    preds = load_predictions_file_cached(res_file, file_signature(res_file))
                    pred_item = next((p for p in preds if p["file_name"] == selected_image), None)
                    if pred_item:
                        try:
                            pred_json = (
                                json.loads(pred_item["prediction"])
                                if isinstance(pred_item["prediction"], str)
                                else pred_item["prediction"]
                            )
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
            st.dataframe(yn_df, width="stretch")

    if v_key == "v2":
        st.divider()
        st.subheader("Y/N Error Breakdown")
        full_results = load_full_results(v_key)
        if full_results:
            default_model = selected_models[0] if selected_models else list(full_results.keys())[0]
            stats_model = st.selectbox(
                "Model for Y/N stats",
                list(full_results.keys()),
                index=list(full_results.keys()).index(default_model),
            )
            top_n = st.slider("Show lowest-accuracy questions", min_value=5, max_value=50, value=20, step=5)
            stats = full_results.get(stats_model, {}).get("yn_question_stats", [])
            if stats:
                df_stats = pd.DataFrame(stats)
                df_stats = df_stats.head(top_n)
                if "match_types" in df_stats.columns:
                    df_stats["match_types"] = df_stats["match_types"].apply(
                        lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else x
                    )
                st.dataframe(df_stats, width="stretch")
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
        default_b = (
            selected_models[1]
            if (selected_models and len(selected_models) > 1)
            else (available_models[1] if len(available_models) > 1 else available_models[0])
        )
        col_cf1, col_cf2 = st.columns(2)
        with col_cf1:
            case_model_a = st.selectbox(
                "Case Finder Model A",
                available_models,
                index=available_models.index(default_a),
                key=f"cf_model_a_{v_key}",
            )
        with col_cf2:
            idx_b = available_models.index(default_b) if default_b in available_models else min(1, len(available_models) - 1)
            case_model_b = st.selectbox("Case Finder Model B", available_models, index=idx_b, key=f"cf_model_b_{v_key}")

        if case_model_a == case_model_b:
            st.info("Choose two different models to mine disagreement and boundary cases.")
        else:
            cases_df = build_case_finder_df(v_key, full_results_for_cases, case_model_a, case_model_b)
            if not cases_df.empty:
                mode = st.selectbox(
                    "Ranking mode",
                    [
                        "Hard cases (both models low)",
                        "Borderline cases (mid quality)",
                        "Disagreement cases (large gap)",
                    ],
                    key=f"cf_mode_{v_key}",
                )
                top_n = st.slider("Rows to show", min_value=5, max_value=50, value=20, step=5, key=f"cf_topn_{v_key}")
                gap_threshold = st.slider(
                    "Minimum quality gap",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.1,
                    step=0.01,
                    key=f"cf_gap_{v_key}",
                )

                filt_df = cases_df[cases_df["abs_gap"] >= gap_threshold].copy()
                if mode == "Hard cases (both models low)":
                    ranked_df = filt_df.sort_values(["avg_quality", "abs_gap"], ascending=[True, False]).head(top_n)
                elif mode == "Borderline cases (mid quality)":
                    filt_df["borderline_distance"] = (filt_df["avg_quality"] - 0.5).abs()
                    ranked_df = filt_df.sort_values(["borderline_distance", "abs_gap"], ascending=[True, False]).head(top_n)
                else:
                    ranked_df = filt_df.sort_values(["abs_gap", "avg_quality"], ascending=[False, True]).head(top_n)

                st.dataframe(ranked_df, width="stretch")
                st.caption("Tip: copy `file_name` from this table, then use the image selector above to inspect details.")
            else:
                st.info("No common per-sample details found for the selected model pair.")
    else:
        st.info("Need at least two models with full report details to use Case Finder.")
