import json
import os
import tempfile
import unittest

from evaluators.evaluator import OCREvaluator
from evaluators.evaluator_v2 import OCREvaluatorV2
from evaluators.metrics import calculate_cer, calculate_ned, calculate_wer
from evaluators.statistical_tests import bootstrap_confidence_interval


def _write_temp_json(data):
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return path


class MetricsTests(unittest.TestCase):
    def test_basic_metrics_exact_match(self):
        self.assertEqual(calculate_cer("ABC", "ABC"), 0.0)
        self.assertEqual(calculate_wer("A B", "A B"), 0.0)
        self.assertEqual(calculate_ned("", ""), 0.0)

    def test_bootstrap_ci_is_reproducible_with_fixed_seed(self):
        data = [0.1, 0.2, 0.3, 0.4, 0.5]
        first = bootstrap_confidence_interval(data, n_bootstrap=500, random_seed=7)
        second = bootstrap_confidence_interval(data, n_bootstrap=500, random_seed=7)
        self.assertEqual(first, second)


class EvaluatorV1Tests(unittest.TestCase):
    def test_empty_prediction_is_still_counted(self):
        gt_path = _write_temp_json([{"file_name": "a.png", "text": "ABC"}])
        self.addCleanup(lambda: os.remove(gt_path))
        evaluator = OCREvaluator(gt_path, normalize=False)

        report = evaluator.evaluate_results([
            {"file_name": "a.png", "prediction": ""}
        ])

        self.assertEqual(report["sample_count"], 1)
        self.assertEqual(report["average_cer"], 1.0)
        self.assertEqual(report["average_wer"], 1.0)


class EvaluatorV2Tests(unittest.TestCase):
    def test_postprocess_enables_key_and_yn_normalization(self):
        gt_path = _write_temp_json([
            {
                "file_name": "a.png",
                "handwriting_text": "abc",
                "yn_options": {"Heart Disease": "Y"},
            }
        ])
        self.addCleanup(lambda: os.remove(gt_path))
        prediction = {
            "file_name": "a.png",
            "prediction": json.dumps(
                {"handwriting_text": "abc", "yn_options": {"心脏病": "YES"}},
                ensure_ascii=False,
            ),
        }

        report = OCREvaluatorV2(gt_path, enable_postprocess=True).evaluate_results([prediction])
        self.assertEqual(report["sample_count"], 1)
        self.assertEqual(report["avg_yn_acc"], 1.0)
        self.assertEqual(report["avg_handwriting_cer"], 0.0)
        self.assertEqual(report["avg_weighted_score"], 1.0)

    def test_no_postprocess_requires_strict_yn_format(self):
        gt_path = _write_temp_json([
            {
                "file_name": "a.png",
                "handwriting_text": "abc",
                "yn_options": {"Heart Disease": "Y"},
            }
        ])
        self.addCleanup(lambda: os.remove(gt_path))
        prediction = {
            "file_name": "a.png",
            "prediction": json.dumps(
                {"handwriting_text": "abc", "yn_options": {"心脏病": "YES"}},
                ensure_ascii=False,
            ),
        }

        report = OCREvaluatorV2(gt_path, enable_postprocess=False).evaluate_results([prediction])
        self.assertEqual(report["sample_count"], 1)
        self.assertEqual(report["avg_yn_acc"], 0.0)
        self.assertEqual(report["avg_handwriting_cer"], 0.0)
        self.assertEqual(report["avg_weighted_score"], 0.5)


if __name__ == "__main__":
    unittest.main()
