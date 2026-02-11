import json
import os
from typing import Dict, List, Optional, Set, Any

DEFAULT_SPLIT_PATH = "data/dataset_split.json"

def load_splits(split_path: Optional[str] = None) -> Optional[Dict[str, List[str]]]:
    path = split_path
    if not path and os.path.exists(DEFAULT_SPLIT_PATH):
        path = DEFAULT_SPLIT_PATH
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        splits = {}
        for key in ("v1", "v2"):
            value = data.get(key, [])
            if isinstance(value, list):
                splits[key] = [str(v) for v in value if v]
            else:
                splits[key] = []
        return splits
    except Exception:
        return None

def get_split_for_version(splits: Optional[Dict[str, List[str]]], version: str) -> Optional[Set[str]]:
    if not splits:
        return None
    files = splits.get(version)
    if not isinstance(files, list) or not files:
        return None
    return set(files)

def filter_gt_data(gt_data: List[Dict[str, Any]], split_set: Optional[Set[str]]) -> List[Dict[str, Any]]:
    if not split_set:
        return gt_data
    return [item for item in gt_data if item.get("file_name") in split_set]

def filter_file_names(file_names: List[str], split_set: Optional[Set[str]]) -> List[str]:
    if not split_set:
        return file_names
    return [name for name in file_names if name in split_set]
