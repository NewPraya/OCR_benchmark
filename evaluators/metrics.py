import difflib
import re
from typing import List, Set

def calculate_cer(prediction: str, ground_truth: str) -> float:
    """
    Character Error Rate (CER).
    """
    if not ground_truth:
        return 1.0 if prediction else 0.0
    
    # Simple edit distance logic using difflib if external libraries aren't available
    # For a production benchmark, use `jiwer` or `Levenshtein`
    s = difflib.SequenceMatcher(None, prediction, ground_truth)
    # This is an approximation. Actual CER is (ins + del + sub) / len(gt)
    # difflib doesn't directly give edit distance easily, so let's use a simple DP for CER.
    return _levenshtein_distance(prediction, ground_truth) / len(ground_truth)

def calculate_wer(prediction: str, ground_truth: str) -> float:
    """
    Word Error Rate (WER).
    For Chinese/Japanese/Korean, we treat each character as a 'word'.
    """
    def tokenize(text):
        if not text: return []
        # Check if contains CJK characters
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            # CJK: treat each char as a token, but keep English words/numbers together
            import re
            # 这里的正则包含了单词、数字以及单个 CJK 字符
            tokens = re.findall(r'[a-zA-Z0-9]+|[\u4e00-\u9fff]', text)
            return tokens
        return text.split()

    pred_words = tokenize(prediction)
    gt_words = tokenize(ground_truth)
    
    if not gt_words:
        return 1.0 if pred_words else 0.0
        
    distance = _levenshtein_distance(pred_words, gt_words)
    return distance / len(gt_words)

def calculate_ned(prediction: str, ground_truth: str) -> float:
    """
    Normalized Edit Distance (NED).
    Normalized version of edit distance, bounded between 0 and 1.
    NED = edit_distance / max(len(pred), len(gt))
    """
    if not ground_truth and not prediction:
        return 0.0
    if not ground_truth:
        return 1.0
    if not prediction:
        return 1.0
    
    edit_dist = _levenshtein_distance(prediction, ground_truth)
    max_len = max(len(prediction), len(ground_truth))
    return edit_dist / max_len if max_len > 0 else 0.0

def calculate_precision_recall(prediction: str, ground_truth: str) -> tuple:
    """
    Calculate character-level precision and recall.
    Returns: (precision, recall)
    """
    if not ground_truth and not prediction:
        return 1.0, 1.0
    if not ground_truth:
        return 0.0, 0.0
    if not prediction:
        return 0.0, 0.0
    
    # Use SequenceMatcher to find matching blocks
    matcher = difflib.SequenceMatcher(None, prediction, ground_truth)
    matches = sum(block.size for block in matcher.get_matching_blocks())
    
    precision = matches / len(prediction) if len(prediction) > 0 else 0.0
    recall = matches / len(ground_truth) if len(ground_truth) > 0 else 0.0
    
    return precision, recall

def calculate_exact_match(prediction: str, ground_truth: str) -> bool:
    """
    Exact Match: checks if prediction exactly matches ground truth.
    Returns True if exact match, False otherwise.
    """
    return prediction == ground_truth

def calculate_bow_f1(prediction: str, ground_truth: str) -> float:
    """
    Bag-of-Words F1 Score.
    Treats text as a bag of words, ignoring order.
    Useful for documents where layout recognition may vary.
    """
    def tokenize(text):
        if not text:
            return set()
        # Check if contains CJK characters
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            # CJK: treat each char as a token, but keep English words/numbers together
            tokens = re.findall(r'[a-zA-Z0-9]+|[\u4e00-\u9fff]', text)
            return set(tokens)
        return set(text.split())
    
    pred_tokens = tokenize(prediction)
    gt_tokens = tokenize(ground_truth)
    
    if not gt_tokens and not pred_tokens:
        return 1.0
    if not gt_tokens or not pred_tokens:
        return 0.0
    
    # Calculate intersection
    common = pred_tokens & gt_tokens
    
    precision = len(common) / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
    recall = len(common) / len(gt_tokens) if len(gt_tokens) > 0 else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def _levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if not s2:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

