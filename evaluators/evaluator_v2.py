import json
from typing import List, Dict, Any
from utils.normalization import normalize_text

class OCREvaluatorV2:
    def __init__(self, ground_truth_path: str):
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            self.gt_data = json.load(f)
        self.gt_dict = {item['file_name']: item for item in self.gt_data}

    def evaluate_results(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_logical_acc = 0.0
        total_disease_acc = 0.0
        total_entity_recall = 0.0
        total_pairing_acc = 0.0
        count = 0
        
        individual_results = []

        for pred in predictions:
            file_name = pred['file_name']
            if file_name in self.gt_dict:
                gt = self.gt_dict[file_name]
                # Extract predicted JSON
                try:
                    pred_data = json.loads(pred['prediction']) if isinstance(pred['prediction'], str) else pred['prediction']
                except:
                    pred_data = {}

                # 1. Logical Value Accuracy (Q1-Q14)
                logical_match = 0
                gt_logical = gt.get('logical_values', {})
                pred_logical = pred_data.get('logical_values', {})
                
                # Normalize predicted keys for robust matching
                norm_pred_logical = {normalize_text(k, strict_semantic=True): v for k, v in pred_logical.items()}
                
                for k, v in gt_logical.items():
                    norm_k = normalize_text(k, strict_semantic=True)
                    if str(norm_pred_logical.get(norm_k)).upper() == str(v).upper():
                        logical_match += 1
                logical_acc = logical_match / len(gt_logical) if gt_logical else 0
                
                # 2. Disease Status Accuracy
                disease_match = 0
                gt_disease = gt.get('disease_status', {})
                pred_disease = pred_data.get('disease_status', {})
                
                # Normalize predicted keys for robust matching
                norm_pred_disease = {normalize_text(k, strict_semantic=True): v for k, v in pred_disease.items()}
                
                for k, v in gt_disease.items():
                    norm_k = normalize_text(k, strict_semantic=True)
                    if str(norm_pred_disease.get(norm_k)).upper() == str(v).upper():
                        disease_match += 1
                disease_acc = disease_match / len(gt_disease) if gt_disease else 0

                # 3. Medical Entity Recall
                gt_entities = [e.lower() for e in gt.get('medical_entities', [])]
                pred_entities_raw = pred_data.get('medical_entities', [])
                if isinstance(pred_entities_raw, list):
                    pred_entities = " ".join([str(e) for e in pred_entities_raw]).lower()
                else:
                    pred_entities = str(pred_entities_raw).lower()
                
                entities_found = 0
                for entity in gt_entities:
                    if entity in pred_entities:
                        entities_found += 1
                entity_recall = entities_found / len(gt_entities) if gt_entities else 0

                # 4. Field Pairing Consistency
                gt_pairings = gt.get('field_pairings', {})
                pred_pairings = pred_data.get('field_pairings', {})
                pairing_match = 0
                for k, v in gt_pairings.items():
                    # Normalize both key and values for comparison
                    norm_k = normalize_text(k, strict_semantic=True)
                    
                    # Try to find the normalized key in pred_pairings
                    pred_v = ""
                    for pk, pv in pred_pairings.items():
                        if normalize_text(pk, strict_semantic=True) == norm_k:
                            pred_v = str(pv)
                            break
                    
                    nv = normalize_text(v, strict_semantic=True)
                    npv = normalize_text(pred_v, strict_semantic=True)
                    
                    if nv and npv:
                        if nv in npv or npv in nv:
                            pairing_match += 1
                pairing_acc = pairing_match / len(gt_pairings) if gt_pairings else 0

                total_logical_acc += logical_acc
                total_disease_acc += disease_acc
                total_entity_recall += entity_recall
                total_pairing_acc += pairing_acc
                count += 1
                
                individual_results.append({
                    "file_name": file_name,
                    "logical_acc": logical_acc,
                    "disease_acc": disease_acc,
                    "entity_recall": entity_recall,
                    "pairing_acc": pairing_acc
                })

        return {
            "avg_logical_acc": total_logical_acc / count if count > 0 else 0,
            "avg_disease_acc": total_disease_acc / count if count > 0 else 0,
            "avg_entity_recall": total_entity_recall / count if count > 0 else 0,
            "avg_pairing_acc": total_pairing_acc / count if count > 0 else 0,
            "sample_count": count,
            "details": individual_results
        }

