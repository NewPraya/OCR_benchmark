import json
import re
from typing import List, Dict, Any
from utils.normalization import normalize_text

class OCREvaluatorV2:
    # Common variations of keys used by different models
    KEY_MAPPING = {
        # Disease Status
        "心脏病": "Heart Disease",
        "高血压": "Hypertension",
        "血病": "Blood Disease",
        "性病": "V.D.",
        "肾病": "Kidney Disease",
        "糖尿病": "Diabetes",
        "甲状腺病": "Thyroid Disease",
        "其他有关健康资料": "Other Medical Problems",
        "结核病": "T.B.",
        "结核病 (肺癆)": "T.B.",
        "羊痫": "Epilepsy",
        "脑充血": "Stroke",
        "肝病": "Liver Disease",
        "黄疸病": "Jaundice",
        "肝炎": "Hepatitis",
        
        # Field Pairings
        "病历": "MEDICAL HISTORY",
        "现在正在接受医药治疗": "Current Medical Treatment",
        "是否服食任何药物": "Medicine Pills or Tablets",
        "入住医院或在医院「门诊部」接受治疗": "Previous Hospital Experience",
        "全身麻醉": "General Anaesthetic",
        "对「盘尼西林」、食物或其他药物有「敏感」": "Allergic to Penicillin or Other Substances",
        "乾草热、湿疹或哮喘": "Hay Fever, Eczema, Asthma",
        "风湿性热病": "Rheumatic Fever",
        "施手术、脱牙或受伤时是否流血不止": "Abnormal Bleeding",
        "类固醇、抗凝血药物或电疗": "Steroid, Anti-coagulant or Irradiation Therapy",
        "黄胆病、肝炎或其他肝病": "Jaundice, Hepatitis or Other Liver Disease",
        "突然晕倒或常觉头晕": "Fainting or Giddiness",
        "曾否患水痘、麻疹、风疹、痄腮等症": "Childhood Diseases",
        "是否怀孕": "Expectant Mother",
        "日期": "Date",
        "药物": "DRUGS"
    }

    def __init__(self, ground_truth_path: str, weights: Dict[str, float] = None):
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            self.gt_data = json.load(f)
        self.gt_dict = {item['file_name']: item for item in self.gt_data}
        
        # Default weights for overall score calculation
        if weights is None:
            self.weights = {
                'logical_values': 0.25,
                'disease_status': 0.20,
                'entity_f1': 0.35,
                'pairing': 0.20
            }
        else:
            self.weights = weights

    def _normalize_key(self, key: str) -> str:
        """Normalize key by stripping, lowering, and mapping common variations."""
        if not key:
            return ""
        key = str(key).strip()
        # Direct mapping check
        if key in self.KEY_MAPPING:
            key = self.KEY_MAPPING[key]
        
        # Semantic normalization (stripping non-alphanumeric and uppercase)
        return normalize_text(key, strict_semantic=True)

    def evaluate_results(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_logical_acc = 0.0
        total_disease_acc = 0.0
        total_entity_recall = 0.0
        total_entity_precision = 0.0
        total_entity_f1 = 0.0
        total_pairing_acc = 0.0
        total_weighted_score = 0.0
        count = 0
        
        individual_results = []
        field_errors = {
            'logical_values': {'correct': 0, 'total': 0},
            'disease_status': {'correct': 0, 'total': 0},
            'medical_entities': {'tp': 0, 'fp': 0, 'fn': 0},
            'field_pairings': {'correct': 0, 'total': 0}
        }

        for pred in predictions:
            file_name = pred['file_name']
            if file_name in self.gt_dict:
                gt = self.gt_dict[file_name]
                # Extract predicted JSON
                try:
                    raw_pred = pred['prediction']
                    if isinstance(raw_pred, str):
                        # Strip markdown code blocks if present
                        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', raw_pred)
                        if json_match:
                            pred_text = json_match.group(1)
                        else:
                            pred_text = raw_pred
                        pred_data = json.loads(pred_text)
                    else:
                        pred_data = raw_pred
                except Exception as e:
                    pred_data = {}

                # 1. Logical Value Accuracy (Q1-Q14)
                logical_match = 0
                gt_logical = gt.get('logical_values', {})
                pred_logical = pred_data.get('logical_values', {})
                
                # Normalize predicted keys for robust matching
                norm_pred_logical = {self._normalize_key(k): v for k, v in pred_logical.items()}
                
                for k, v in gt_logical.items():
                    norm_k = self._normalize_key(k)
                    pred_v = norm_pred_logical.get(norm_k)
                    
                    # Fuzzy match for Y/N values
                    is_correct = False
                    if pred_v is not None:
                        pv = str(pred_v).upper().strip()
                        gv = str(v).upper().strip()
                        if pv == gv:
                            is_correct = True
                        elif gv == "Y" and pv in ["YES", "T", "TRUE", "CHECKED", "V"]:
                            is_correct = True
                        elif gv == "N" and pv in ["NO", "F", "FALSE", "UNCHECKED", "X"]:
                            is_correct = True
                            
                    if is_correct:
                        logical_match += 1
                    field_errors['logical_values']['total'] += 1
                logical_acc = logical_match / len(gt_logical) if gt_logical else 0
                field_errors['logical_values']['correct'] += logical_match
                
                # 2. Disease Status Accuracy
                disease_match = 0
                gt_disease = gt.get('disease_status', {})
                pred_disease = pred_data.get('disease_status', {})
                
                # Normalize predicted keys for robust matching
                norm_pred_disease = {self._normalize_key(k): v for k, v in pred_disease.items()}
                
                for k, v in gt_disease.items():
                    norm_k = self._normalize_key(k)
                    pred_v = norm_pred_disease.get(norm_k)
                    
                    # Fuzzy match for Y/N values
                    is_correct = False
                    if pred_v is not None:
                        pv = str(pred_v).upper().strip()
                        gv = str(v).upper().strip()
                        if pv == gv:
                            is_correct = True
                        elif gv == "Y" and pv in ["YES", "T", "TRUE", "CHECKED", "V"]:
                            is_correct = True
                        elif gv == "N" and pv in ["NO", "F", "FALSE", "UNCHECKED", "X"]:
                            is_correct = True
                            
                    if is_correct:
                        disease_match += 1
                    field_errors['disease_status']['total'] += 1
                disease_acc = disease_match / len(gt_disease) if gt_disease else 0
                field_errors['disease_status']['correct'] += disease_match

                # 3. Medical Entity Precision, Recall, and F1
                gt_entities = [normalize_text(e, remove_punctuation=True) for e in gt.get('medical_entities', [])]
                pred_entities_raw = pred_data.get('medical_entities', [])
                
                # Normalize predicted entities to list of strings
                if isinstance(pred_entities_raw, list):
                    pred_entities = [normalize_text(str(e), remove_punctuation=True) for e in pred_entities_raw]
                else:
                    pred_entities = [normalize_text(str(pred_entities_raw), remove_punctuation=True)] if pred_entities_raw else []
                
                # Calculate True Positives, False Positives, False Negatives
                true_positives = 0
                for gt_entity in gt_entities:
                    if not gt_entity: continue
                    # Check if this entity appears in any predicted entity
                    if any(gt_entity in pred_entity or pred_entity in gt_entity for pred_entity in pred_entities if pred_entity):
                        true_positives += 1
                
                false_negatives = len([e for e in gt_entities if e]) - true_positives
                
                # Count false positives: predicted entities not matching any GT entity
                false_positives = 0
                for pred_entity in pred_entities:
                    if not pred_entity: continue
                    if not any(gt_entity in pred_entity or pred_entity in gt_entity for gt_entity in gt_entities if gt_entity):
                        false_positives += 1
                
                # Calculate metrics
                entity_recall = true_positives / len([e for e in gt_entities if e]) if any(gt_entities) else 0
                entity_precision = true_positives / len([e for e in pred_entities if e]) if any(pred_entities) else 0
                
                if entity_precision + entity_recall > 0:
                    entity_f1 = 2 * (entity_precision * entity_recall) / (entity_precision + entity_recall)
                else:
                    entity_f1 = 0.0
                
                # Update field errors for global statistics
                field_errors['medical_entities']['tp'] += true_positives
                field_errors['medical_entities']['fp'] += false_positives
                field_errors['medical_entities']['fn'] += false_negatives

                # 4. Field Pairing Consistency
                gt_pairings = gt.get('field_pairings', {})
                pred_pairings = pred_data.get('field_pairings', {})
                pairing_match = 0
                for k, v in gt_pairings.items():
                    # Normalize both key and values for comparison
                    norm_k = self._normalize_key(k)
                    
                    # Try to find the normalized key in pred_pairings
                    pred_v = ""
                    for pk, pv in pred_pairings.items():
                        if self._normalize_key(pk) == norm_k:
                            pred_v = str(pv)
                            break
                    
                    nv = normalize_text(v, strict_semantic=True)
                    npv = normalize_text(pred_v, strict_semantic=True)
                    
                    if nv and npv:
                        if nv in npv or npv in nv:
                            pairing_match += 1
                    field_errors['field_pairings']['total'] += 1
                pairing_acc = pairing_match / len(gt_pairings) if gt_pairings else 0
                field_errors['field_pairings']['correct'] += pairing_match

                # Calculate weighted overall score
                weighted_score = (
                    logical_acc * self.weights['logical_values'] +
                    disease_acc * self.weights['disease_status'] +
                    entity_f1 * self.weights['entity_f1'] +
                    pairing_acc * self.weights['pairing']
                )
                
                total_logical_acc += logical_acc
                total_disease_acc += disease_acc
                total_entity_recall += entity_recall
                total_entity_precision += entity_precision
                total_entity_f1 += entity_f1
                total_pairing_acc += pairing_acc
                total_weighted_score += weighted_score
                count += 1
                
                individual_results.append({
                    "file_name": file_name,
                    "logical_acc": logical_acc,
                    "disease_acc": disease_acc,
                    "entity_precision": entity_precision,
                    "entity_recall": entity_recall,
                    "entity_f1": entity_f1,
                    "pairing_acc": pairing_acc,
                    "weighted_score": weighted_score
                })

        # Calculate per-field accuracies
        field_analysis = {}
        for field_name, stats in field_errors.items():
            if field_name == 'medical_entities':
                # For entities, calculate P/R/F1
                tp = stats['tp']
                fp = stats['fp']
                fn = stats['fn']
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                field_analysis[field_name] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'tp': tp,
                    'fp': fp,
                    'fn': fn
                }
            else:
                # For other fields, calculate accuracy
                accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                field_analysis[field_name] = {
                    'accuracy': accuracy,
                    'correct': stats['correct'],
                    'total': stats['total']
                }
        
        return {
            "avg_logical_acc": total_logical_acc / count if count > 0 else 0,
            "avg_disease_acc": total_disease_acc / count if count > 0 else 0,
            "avg_entity_precision": total_entity_precision / count if count > 0 else 0,
            "avg_entity_recall": total_entity_recall / count if count > 0 else 0,
            "avg_entity_f1": total_entity_f1 / count if count > 0 else 0,
            "avg_pairing_acc": total_pairing_acc / count if count > 0 else 0,
            "avg_weighted_score": total_weighted_score / count if count > 0 else 0,
            "sample_count": count,
            "field_analysis": field_analysis,
            "weights": self.weights,
            "details": individual_results
        }

