import difflib

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

