import json
from typing import List, Dict, Any
from evaluators.metrics import (
    calculate_cer, calculate_wer, calculate_ned, 
    calculate_precision_recall, calculate_exact_match, calculate_bow_f1
)
from utils.normalization import normalize_text

class OCREvaluator:
    def __init__(self, ground_truth_path: str, normalize: bool = True):
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            self.gt_data = json.load(f)
        self.normalize = normalize
        # Convert to dict for easy lookup
        self.gt_dict = {item['file_name']: item['text'] for item in self.gt_data}

    def evaluate_results(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_cer = 0.0
        total_wer = 0.0
        total_ned = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_bow_f1 = 0.0
        exact_matches = 0
        count = 0
        
        individual_results = []

        for pred in predictions:
            file_name = pred['file_name']
            if file_name in self.gt_dict:
                gt_text = self.gt_dict[file_name]
                pred_text = pred.get('prediction', "")
                
                if self.normalize:
                    # 使用更严格的归一化，去除标点干扰
                    gt_text = normalize_text(gt_text, remove_punctuation=True)
                    pred_text = normalize_text(pred_text, remove_punctuation=True)

                # 如果归一化后为空，跳过或设置错误率
                if not gt_text:
                    cer = 0.0 if not pred_text else 1.0
                    wer = 0.0 if not pred_text else 1.0
                    ned = 0.0 if not pred_text else 1.0
                    precision, recall = (1.0, 1.0) if not pred_text else (0.0, 0.0)
                    bow_f1 = 1.0 if not pred_text else 0.0
                    exact_match = not pred_text
                else:
                    cer = calculate_cer(pred_text, gt_text)
                    wer = calculate_wer(pred_text, gt_text)
                    ned = calculate_ned(pred_text, gt_text)
                    precision, recall = calculate_precision_recall(pred_text, gt_text)
                    bow_f1 = calculate_bow_f1(pred_text, gt_text)
                    exact_match = calculate_exact_match(pred_text, gt_text)
                
                total_cer += cer
                total_wer += wer
                total_ned += ned
                total_precision += precision
                total_recall += recall
                total_bow_f1 += bow_f1
                if exact_match:
                    exact_matches += 1
                count += 1
                
                individual_results.append({
                    "file_name": file_name,
                    "cer": cer,
                    "wer": wer,
                    "ned": ned,
                    "precision": precision,
                    "recall": recall,
                    "bow_f1": bow_f1,
                    "exact_match": exact_match
                })

        avg_cer = total_cer / count if count > 0 else 0
        avg_wer = total_wer / count if count > 0 else 0
        avg_ned = total_ned / count if count > 0 else 0
        avg_precision = total_precision / count if count > 0 else 0
        avg_recall = total_recall / count if count > 0 else 0
        avg_bow_f1 = total_bow_f1 / count if count > 0 else 0
        exact_match_acc = exact_matches / count if count > 0 else 0

        return {
            "average_cer": avg_cer,
            "average_wer": avg_wer,
            "average_ned": avg_ned,
            "average_precision": avg_precision,
            "average_recall": avg_recall,
            "average_bow_f1": avg_bow_f1,
            "exact_match_accuracy": exact_match_acc,
            "sample_count": count,
            "details": individual_results
        }

