import json
import os

import pandas as pd
import streamlit as st

from dashboard.utils import file_signature


def _summary_file_signatures(v_key: str):
    files = [
        f"results/multirun/leaderboard_{v_key}.json",
        f"results/multirun/per_run_{v_key}.json",
        f"results/multirun/leaderboard_std_{v_key}.json",
        f"results/multirun/summary_meta_{v_key}.json",
    ]
    return tuple(file_signature(path) for path in files)


@st.cache_data(show_spinner=False)
def _load_multirun_precomputed_cached(v_key: str, file_sigs):
    leaderboard_path = f"results/multirun/leaderboard_{v_key}.json"
    per_run_path = f"results/multirun/per_run_{v_key}.json"
    std_path = f"results/multirun/leaderboard_std_{v_key}.json"
    meta_path = f"results/multirun/summary_meta_{v_key}.json"

    if not os.path.exists(leaderboard_path) or not os.path.exists(per_run_path):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    try:
        with open(leaderboard_path, "r") as f:
            leaderboard_rows = json.load(f)
        with open(per_run_path, "r") as f:
            per_run_rows = json.load(f)

        std_rows = []
        meta = {}

        if os.path.exists(std_path):
            with open(std_path, "r") as f:
                std_rows = json.load(f)

        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)

        leaderboard_df = pd.DataFrame(leaderboard_rows)
        per_run_df = pd.DataFrame(per_run_rows)
        std_df = pd.DataFrame(std_rows)

        return leaderboard_df, per_run_df, std_df, meta if isinstance(meta, dict) else {}
    except Exception:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}


def load_multirun_precomputed(v_key: str):
    file_sigs = _summary_file_signatures(v_key)
    return _load_multirun_precomputed_cached(v_key, file_sigs)
