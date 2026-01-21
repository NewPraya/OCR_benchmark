"""
Schema-agnostic evaluator that works with any document schema.
"""

import json
from typing import List, Dict, Any
from utils.normalization import normalize_text
from schemas.schema_base import DocumentSchema, SchemaLoader


class SchemaBasedEvaluator:
    """
    Generic evaluator that works with any DocumentSchema.
    Evaluation logic is driven by schema configuration.
    """
    
    # Common variations of keys for robust mapping (especially for the medical form)
    KEY_MAPPING = {
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

    def __init__(self, ground_truth_path: str, schema: DocumentSchema):
        """
        Initialize evaluator with ground truth and schema.
        
        Args:
            ground_truth_path: Path to ground truth JSON file
            schema: DocumentSchema instance defining the document structure
        """
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            self.gt_data = json.load(f)
        self.gt_dict = {item['file_name']: item for item in self.gt_data}
        self.schema = schema
    
    def evaluate_results(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate predictions against ground truth using schema-defined metrics.
        
        Args:
            predictions: List of prediction dictionaries
            
        Returns:
            Dictionary containing evaluation results
        """
        field_totals = {field.name: 0.0 for field in self.schema.fields}
        overall_scores = []
        count = 0
        
        individual_results = []
        field_errors = self._initialize_field_errors()
        
        for pred in predictions:
            file_name = pred['file_name']
            if file_name not in self.gt_dict:
                continue
                
            gt = self.gt_dict[file_name]
            
            # Extract predicted JSON
            try:
                pred_data = json.loads(pred['prediction']) if isinstance(pred['prediction'], str) else pred['prediction']
            except:
                pred_data = {}
            
            # Evaluate each field according to schema
            field_scores = {}
            for field in self.schema.fields:
                score = self._evaluate_field(
                    field, 
                    gt.get(field.name), 
                    pred_data.get(field.name),
                    field_errors
                )
                field_scores[field.name] = score
                field_totals[field.name] += score
            
            # Calculate weighted overall score
            weighted_score = sum(
                field_scores[field.name] * self.schema.get_weight(field.name)
                for field in self.schema.fields
            )
            
            overall_scores.append(weighted_score)
            count += 1
            
            # Store individual result
            result = {"file_name": file_name}
            result.update(field_scores)
            result["weighted_score"] = weighted_score
            individual_results.append(result)
        
        # Calculate averages
        avg_results = {
            f"avg_{field.name}": field_totals[field.name] / count if count > 0 else 0
            for field in self.schema.fields
        }
        avg_results["avg_weighted_score"] = sum(overall_scores) / count if count > 0 else 0
        avg_results["sample_count"] = count
        avg_results["details"] = individual_results
        avg_results["schema_name"] = self.schema.schema_name
        avg_results["weights"] = self.schema.weights
        
        # Add field analysis
        avg_results["field_analysis"] = self._analyze_field_errors(field_errors)
        
        return avg_results
    
    def _initialize_field_errors(self) -> Dict[str, Dict[str, Any]]:
        """Initialize field error tracking structure."""
        field_errors = {}
        for field in self.schema.fields:
            if field.evaluation == 'f1':
                field_errors[field.name] = {'tp': 0, 'fp': 0, 'fn': 0}
            else:
                field_errors[field.name] = {'correct': 0, 'total': 0}
        return field_errors
    
    def _evaluate_field(self, field, gt_value, pred_value, field_errors) -> float:
        """
        Evaluate a single field based on its evaluation type.
        
        Args:
            field: FieldConfig object
            gt_value: Ground truth value
            pred_value: Predicted value
            field_errors: Dictionary to track field-level errors
            
        Returns:
            Score for this field (0.0 to 1.0)
        """
        if field.evaluation == 'accuracy':
            return self._evaluate_accuracy(field.name, gt_value, pred_value, field_errors)
        elif field.evaluation == 'f1':
            return self._evaluate_f1(field.name, gt_value, pred_value, field_errors)
        elif field.evaluation == 'pairing':
            return self._evaluate_pairing(field.name, gt_value, pred_value, field_errors)
        elif field.evaluation == 'exact_match':
            return self._evaluate_exact_match(field.name, gt_value, pred_value, field_errors)
        else:
            # Default to accuracy
            return self._evaluate_accuracy(field.name, gt_value, pred_value, field_errors)
    
    def _normalize_key(self, key: str) -> str:
        """Normalize key by stripping, mapping common variations, and semantic normalization."""
        if not key:
            return ""
        key = str(key).strip()
        # Direct mapping check
        if key in self.KEY_MAPPING:
            key = self.KEY_MAPPING[key]
        
        # Semantic normalization (stripping non-alphanumeric and uppercase)
        return normalize_text(key, strict_semantic=True)

    def _evaluate_accuracy(self, field_name, gt_value, pred_value, field_errors) -> float:
        """Evaluate categorical dictionary accuracy (e.g., Y/N fields)."""
        if not gt_value or not isinstance(gt_value, dict):
            return 0.0
        
        pred_value = pred_value or {}
        
        # Normalize keys for robust matching
        norm_pred = {self._normalize_key(k): v for k, v in pred_value.items()}
        
        matches = 0
        for k, v in gt_value.items():
            norm_k = self._normalize_key(k)
            pred_v = norm_pred.get(norm_k)
            
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
                matches += 1
            field_errors[field_name]['total'] += 1
        
        field_errors[field_name]['correct'] += matches
        return matches / len(gt_value) if gt_value else 0.0
    
    def _evaluate_f1(self, field_name, gt_value, pred_value, field_errors) -> float:
        """Evaluate entity list using F1 score."""
        if not gt_value or not isinstance(gt_value, list):
            return 0.0
        
        gt_entities = [str(e).lower().strip() for e in gt_value]
        
        # Normalize predicted entities
        if isinstance(pred_value, list):
            pred_entities = [str(e).lower().strip() for e in pred_value]
        elif pred_value:
            pred_entities = [str(pred_value).lower().strip()]
        else:
            pred_entities = []
        
        # Calculate TP, FP, FN
        true_positives = 0
        for gt_entity in gt_entities:
            if any(gt_entity in pred or pred in gt_entity for pred in pred_entities):
                true_positives += 1
        
        false_negatives = len(gt_entities) - true_positives
        false_positives = sum(
            1 for pred in pred_entities
            if not any(gt in pred or pred in gt for gt in gt_entities)
        )
        
        # Update field errors
        field_errors[field_name]['tp'] += true_positives
        field_errors[field_name]['fp'] += false_positives
        field_errors[field_name]['fn'] += false_negatives
        
        # Calculate F1
        precision = true_positives / len(pred_entities) if pred_entities else 0
        recall = true_positives / len(gt_entities) if gt_entities else 0
        
        if precision + recall > 0:
            return 2 * (precision * recall) / (precision + recall)
        return 0.0
    
    def _evaluate_pairing(self, field_name, gt_value, pred_value, field_errors) -> float:
        """Evaluate field-value pairing (key-value matching with fuzzy text comparison)."""
        if not gt_value or not isinstance(gt_value, dict):
            return 0.0
        
        pred_value = pred_value or {}
        matches = 0
        
        # Normalize predicted keys for robust matching
        norm_pred = {self._normalize_key(k): v for k, v in pred_value.items()}
        
        for k, v in gt_value.items():
            norm_k = self._normalize_key(k)
            
            # Find matching value in normalized predictions
            pred_v = norm_pred.get(norm_k, "")
            
            # Compare values with fuzzy matching
            nv = normalize_text(str(v), strict_semantic=True)
            npv = normalize_text(str(pred_v), strict_semantic=True)
            
            if nv and npv and (nv in npv or npv in nv):
                matches += 1
            field_errors[field_name]['total'] += 1
        
        field_errors[field_name]['correct'] += matches
        return matches / len(gt_value) if gt_value else 0.0
    
    def _evaluate_exact_match(self, field_name, gt_value, pred_value, field_errors) -> float:
        """Evaluate exact match (e.g., for numerical fields)."""
        if not gt_value or not isinstance(gt_value, dict):
            return 0.0
        
        pred_value = pred_value or {}
        matches = 0
        
        for k, v in gt_value.items():
            if k in pred_value and pred_value[k] == v:
                matches += 1
            field_errors[field_name]['total'] += 1
        
        field_errors[field_name]['correct'] += matches
        return matches / len(gt_value) if gt_value else 0.0
    
    def _analyze_field_errors(self, field_errors) -> Dict[str, Dict[str, Any]]:
        """Analyze field-level errors for detailed reporting."""
        analysis = {}
        
        for field_name, stats in field_errors.items():
            if 'tp' in stats:
                # F1-based field
                tp = stats['tp']
                fp = stats['fp']
                fn = stats['fn']
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                analysis[field_name] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'tp': tp,
                    'fp': fp,
                    'fn': fn
                }
            else:
                # Accuracy-based field
                accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                analysis[field_name] = {
                    'accuracy': accuracy,
                    'correct': stats['correct'],
                    'total': stats['total']
                }
        
        return analysis


def create_evaluator_from_schema(ground_truth_path: str, schema_path: str) -> SchemaBasedEvaluator:
    """
    Factory function to create an evaluator from a schema file.
    
    Args:
        ground_truth_path: Path to ground truth JSON
        schema_path: Path to schema YAML file
        
    Returns:
        SchemaBasedEvaluator instance
    """
    schema = SchemaLoader.load_schema(schema_path)
    return SchemaBasedEvaluator(ground_truth_path, schema)
