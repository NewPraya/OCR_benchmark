import json
import re
from typing import List, Dict, Any, Iterable, Tuple
from evaluators.metrics import calculate_cer, calculate_wer, calculate_ned
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
        "羊癫": "Epilepsy",
        "羊癲": "Epilepsy",
        "羊癇": "Epilepsy",
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

    YN_TRUE = {"Y", "YES", "T", "TRUE", "CHECKED", "V", "✓", "✔"}
    YN_FALSE = {"N", "NO", "F", "FALSE", "UNCHECKED", "X", "✗", "✘"}

    def __init__(self, ground_truth_path: str, weights: Dict[str, float] = None):
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            self.gt_data = json.load(f)
        self.gt_dict = {item['file_name']: item for item in self.gt_data}
        
        # Default weights for overall score calculation
        if weights is None:
            self.weights = {
                'yn_accuracy': 0.5,
                'handwriting_score': 0.5
            }
        else:
            self.weights = weights

    def _normalize_key(self, key: str) -> str:
        """Normalize key by stripping, lowering, and mapping common variations."""
        if not key:
            return ""
        key = str(key).strip()
        if key in self.KEY_MAPPING:
            key = self.KEY_MAPPING[key]
        return normalize_text(key, strict_semantic=True)

    def _strip_leading_enum(self, key: str) -> str:
        """Strip leading enumeration like '1.', 'Q1', '1)'."""
        return re.sub(r'^\s*(?:[A-Z]\d+|Q\d+|\d+)[\s\.:、\)\-]*', '', key, flags=re.IGNORECASE)

    def _normalize_key_variants(self, key: str) -> Tuple[str, str]:
        """Return (full_norm, ascii_norm) for fuzzy key matching."""
        if not key:
            return ("", "")
        key = str(key).strip()
        if key in self.KEY_MAPPING:
            key = self.KEY_MAPPING[key]
        else:
            for k, v in self.KEY_MAPPING.items():
                if k and k in key:
                    key = key.replace(k, v)
        key = self._strip_leading_enum(key)
        full_norm = normalize_text(key, remove_punctuation=True, strict_semantic=False)
        full_norm = full_norm.replace(" ", "")
        ascii_norm = re.sub(r'[^A-Z0-9]', '', full_norm)
        return (full_norm, ascii_norm)

    def _find_pred_value(self, gt_key: str, pred_entries: Iterable[Tuple[str, Any, Tuple[str, str]]]) -> Any:
        """Find best matching pred value for a GT key using fuzzy matching."""
        gt_full, gt_ascii = self._normalize_key_variants(gt_key)
        if not gt_full and not gt_ascii:
            return None
        # 1) Exact match on normalized variants
        for _, value, (p_full, p_ascii) in pred_entries:
            if gt_full and p_full and gt_full == p_full:
                return value
            if gt_ascii and p_ascii and gt_ascii == p_ascii:
                return value
        # 2) Substring match on ASCII (guard with length to avoid collisions)
        if gt_ascii and len(gt_ascii) >= 4:
            for _, value, (_, p_ascii) in pred_entries:
                if p_ascii and (gt_ascii in p_ascii or p_ascii in gt_ascii):
                    return value
        # 3) Substring match on full normalized (for CJK/mixed labels)
        if gt_full and len(gt_full) >= 4:
            for _, value, (p_full, _) in pred_entries:
                if p_full and (gt_full in p_full or p_full in gt_full):
                    return value
        return None

    def _normalize_yn(self, value: Any) -> str:
        """Normalize a Y/N-like value to 'Y' or 'N'."""
        if value is None:
            return ""
        val = str(value).strip().upper()
        if val in self.YN_TRUE:
            return "Y"
        if val in self.YN_FALSE:
            return "N"
        return val if val in {"Y", "N"} else ""

    def _parse_prediction(self, raw_pred: Any) -> Dict[str, Any]:
        """Parse prediction into a dict, stripping markdown if needed."""
        if isinstance(raw_pred, dict):
            return raw_pred
        if not isinstance(raw_pred, str):
            return {}
        
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', raw_pred)
        pred_text = json_match.group(1) if json_match else raw_pred
        try:
            return json.loads(pred_text)
        except Exception:
            return {}

    def _build_gt_handwriting_text(self, gt: Dict[str, Any]) -> str:
        """Combine GT handwritten content into a single text block."""
        if gt.get("handwriting_text") is not None:
            return str(gt.get("handwriting_text", ""))
        chunks = []
        for entity in gt.get('medical_entities', []):
            if entity:
                chunks.append(str(entity))
        for _, value in gt.get('field_pairings', {}).items():
            if value:
                chunks.append(str(value))
        return "\n".join(chunks)

    def _extract_handwriting_from_pred(self, pred_data: Dict[str, Any]) -> str:
        """Extract handwriting text from prediction with fallbacks."""
        for key in ("handwriting_text", "handwritten_text", "handwriting"):
            if key in pred_data and pred_data[key] is not None:
                return str(pred_data[key])
        # Backward compatibility for structured outputs
        chunks = []
        pred_entities = pred_data.get("medical_entities", [])
        if isinstance(pred_entities, list):
            chunks.extend([str(e) for e in pred_entities if e])
        elif pred_entities:
            chunks.append(str(pred_entities))
        pred_pairings = pred_data.get("field_pairings", {})
        if isinstance(pred_pairings, dict):
            chunks.extend([str(v) for v in pred_pairings.values() if v])
        return "\n".join(chunks)

    def _extract_yn_options_from_pred(self, pred_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Y/N options dict from prediction with fallbacks."""
        for key in ("yn_options", "logical_values", "disease_status", "options"):
            value = pred_data.get(key)
            if isinstance(value, dict):
                return value
        return {}

    def evaluate_results(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_yn_acc = 0.0
        total_handwriting_cer = 0.0
        total_handwriting_wer = 0.0
        total_handwriting_ned = 0.0
        total_weighted_score = 0.0
        count = 0
        
        individual_results = []
        field_errors = {
            'yn_options': {'correct': 0, 'total': 0}
        }

        for pred in predictions:
            file_name = pred['file_name']
            if file_name not in self.gt_dict:
                continue
            
            gt = self.gt_dict[file_name]
            pred_data = self._parse_prediction(pred.get('prediction', {}))

            # Build GT targets (new format first, fallback to legacy fields)
            if isinstance(gt.get("yn_options"), dict):
                gt_yn = gt.get("yn_options", {})
            else:
                gt_yn = {}
                gt_yn.update(gt.get('logical_values', {}))
                gt_yn.update(gt.get('disease_status', {}))
            gt_handwriting = self._build_gt_handwriting_text(gt)

            # Extract predictions
            pred_yn_raw = self._extract_yn_options_from_pred(pred_data)
            pred_handwriting = self._extract_handwriting_from_pred(pred_data)

            # Precompute normalized key variants for fuzzy matching
            pred_entries = []
            for k, v in pred_yn_raw.items():
                pred_entries.append((k, v, self._normalize_key_variants(k)))

            # 1) Y/N accuracy
            yn_match = 0
            for k, v in gt_yn.items():
                pred_v = self._find_pred_value(k, pred_entries)
                gv = self._normalize_yn(v)
                pv = self._normalize_yn(pred_v)
                if gv and pv and gv == pv:
                    yn_match += 1
                field_errors['yn_options']['total'] += 1
            yn_acc = yn_match / len(gt_yn) if gt_yn else 0.0
            field_errors['yn_options']['correct'] += yn_match

            # 2) Handwriting text metrics
            gt_text = normalize_text(gt_handwriting, remove_punctuation=True)
            pred_text = normalize_text(pred_handwriting, remove_punctuation=True)

            if not gt_text:
                cer = 0.0 if not pred_text else 1.0
                wer = 0.0 if not pred_text else 1.0
                ned = 0.0 if not pred_text else 1.0
            else:
                cer = calculate_cer(pred_text, gt_text)
                wer = calculate_wer(pred_text, gt_text)
                ned = calculate_ned(pred_text, gt_text)

            handwriting_score = max(0.0, 1.0 - cer)
            weighted_score = (
                yn_acc * self.weights['yn_accuracy'] +
                handwriting_score * self.weights['handwriting_score']
            )

            total_yn_acc += yn_acc
            total_handwriting_cer += cer
            total_handwriting_wer += wer
            total_handwriting_ned += ned
            total_weighted_score += weighted_score
            count += 1

            individual_results.append({
                "file_name": file_name,
                "yn_acc": yn_acc,
                "handwriting_cer": cer,
                "handwriting_wer": wer,
                "handwriting_ned": ned,
                "weighted_score": weighted_score
            })

        yn_accuracy = field_errors['yn_options']['correct'] / field_errors['yn_options']['total'] if field_errors['yn_options']['total'] > 0 else 0
        field_analysis = {
            "yn_options": {
                "accuracy": yn_accuracy,
                "correct": field_errors['yn_options']['correct'],
                "total": field_errors['yn_options']['total']
            }
        }

        return {
            "avg_yn_acc": total_yn_acc / count if count > 0 else 0,
            "avg_handwriting_cer": total_handwriting_cer / count if count > 0 else 0,
            "avg_handwriting_wer": total_handwriting_wer / count if count > 0 else 0,
            "avg_handwriting_ned": total_handwriting_ned / count if count > 0 else 0,
            "avg_weighted_score": total_weighted_score / count if count > 0 else 0,
            "sample_count": count,
            "field_analysis": field_analysis,
            "weights": self.weights,
            "details": individual_results
        }

