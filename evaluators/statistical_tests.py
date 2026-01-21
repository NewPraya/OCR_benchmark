"""
Statistical analysis tools for OCR benchmark results.
Provides confidence intervals and significance testing for model comparisons.
"""

import numpy as np
from typing import List, Dict, Tuple, Any
from scipy import stats
import warnings


def bootstrap_confidence_interval(
    data: List[float], 
    confidence_level: float = 0.95, 
    n_bootstrap: int = 10000,
    statistic_fn=np.mean
) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence interval for a metric.
    
    Args:
        data: List of metric values
        confidence_level: Confidence level (default: 0.95 for 95% CI)
        n_bootstrap: Number of bootstrap samples
        statistic_fn: Function to compute statistic (default: mean)
        
    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound)
    """
    if not data:
        return 0.0, 0.0, 0.0
    
    data = np.array(data)
    n = len(data)
    
    # Generate bootstrap samples
    bootstrap_samples = np.random.choice(data, size=(n_bootstrap, n), replace=True)
    bootstrap_stats = np.apply_along_axis(statistic_fn, axis=1, arr=bootstrap_samples)
    
    # Calculate point estimate
    point_estimate = statistic_fn(data)
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(bootstrap_stats, lower_percentile)
    ci_upper = np.percentile(bootstrap_stats, upper_percentile)
    
    return point_estimate, ci_lower, ci_upper


def paired_t_test(
    model1_scores: List[float], 
    model2_scores: List[float],
    alternative: str = 'two-sided'
) -> Dict[str, Any]:
    """
    Perform paired t-test to compare two models.
    
    Args:
        model1_scores: Scores from model 1 (one per sample)
        model2_scores: Scores from model 2 (one per sample)
        alternative: 'two-sided', 'less', or 'greater'
        
    Returns:
        Dictionary with test results
    """
    if len(model1_scores) != len(model2_scores):
        raise ValueError("Both models must have the same number of samples")
    
    if len(model1_scores) < 2:
        return {
            'test': 'paired_t_test',
            'statistic': None,
            'p_value': None,
            'significant': False,
            'message': 'Insufficient samples for t-test (need at least 2)'
        }
    
    # Perform paired t-test
    statistic, p_value = stats.ttest_rel(model1_scores, model2_scores, alternative=alternative)
    
    # Check significance at alpha=0.05
    is_significant = p_value < 0.05
    
    # Calculate effect size (Cohen's d for paired samples)
    differences = np.array(model1_scores) - np.array(model2_scores)
    cohens_d = np.mean(differences) / np.std(differences, ddof=1) if np.std(differences, ddof=1) > 0 else 0
    
    return {
        'test': 'paired_t_test',
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': is_significant,
        'alpha': 0.05,
        'cohens_d': float(cohens_d),
        'mean_difference': float(np.mean(differences)),
        'interpretation': _interpret_significance(p_value, is_significant)
    }


def wilcoxon_signed_rank_test(
    model1_scores: List[float],
    model2_scores: List[float],
    alternative: str = 'two-sided'
) -> Dict[str, Any]:
    """
    Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test).
    More robust to outliers and non-normal distributions.
    
    Args:
        model1_scores: Scores from model 1 (one per sample)
        model2_scores: Scores from model 2 (one per sample)
        alternative: 'two-sided', 'less', or 'greater'
        
    Returns:
        Dictionary with test results
    """
    if len(model1_scores) != len(model2_scores):
        raise ValueError("Both models must have the same number of samples")
    
    if len(model1_scores) < 3:
        return {
            'test': 'wilcoxon_signed_rank',
            'statistic': None,
            'p_value': None,
            'significant': False,
            'message': 'Insufficient samples for Wilcoxon test (need at least 3)'
        }
    
    # Suppress warnings for zero differences
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = stats.wilcoxon(
                model1_scores, 
                model2_scores, 
                alternative=alternative,
                zero_method='wilcox'
            )
            statistic, p_value = result.statistic, result.pvalue
        except ValueError as e:
            return {
                'test': 'wilcoxon_signed_rank',
                'statistic': None,
                'p_value': None,
                'significant': False,
                'message': f'Test failed: {str(e)}'
            }
    
    is_significant = p_value < 0.05
    
    differences = np.array(model1_scores) - np.array(model2_scores)
    
    return {
        'test': 'wilcoxon_signed_rank',
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': is_significant,
        'alpha': 0.05,
        'median_difference': float(np.median(differences)),
        'interpretation': _interpret_significance(p_value, is_significant)
    }


def compare_models(
    model1_results: Dict[str, Any],
    model2_results: Dict[str, Any],
    metric_name: str = 'weighted_score',
    use_parametric: bool = True
) -> Dict[str, Any]:
    """
    Compare two models with statistical significance testing.
    
    Args:
        model1_results: Results dictionary from model 1 (with 'details' key)
        model2_results: Results dictionary from model 2 (with 'details' key)
        metric_name: Name of metric to compare (e.g., 'weighted_score', 'cer', 'wer')
        use_parametric: If True, use t-test; if False, use Wilcoxon test
        
    Returns:
        Dictionary with comparison results and statistics
    """
    # Extract scores for each sample
    model1_details = model1_results.get('details', [])
    model2_details = model2_results.get('details', [])
    
    # Match samples by file_name
    model1_dict = {item['file_name']: item for item in model1_details}
    model2_dict = {item['file_name']: item for item in model2_details}
    
    common_files = set(model1_dict.keys()) & set(model2_dict.keys())
    
    if not common_files:
        return {
            'error': 'No common samples found between models',
            'model1_samples': len(model1_details),
            'model2_samples': len(model2_details)
        }
    
    model1_scores = [model1_dict[f].get(metric_name, 0) for f in sorted(common_files)]
    model2_scores = [model2_dict[f].get(metric_name, 0) for f in sorted(common_files)]
    
    # Perform significance test
    if use_parametric:
        test_result = paired_t_test(model1_scores, model2_scores)
    else:
        test_result = wilcoxon_signed_rank_test(model1_scores, model2_scores)
    
    # Calculate confidence intervals for both models
    m1_mean, m1_lower, m1_upper = bootstrap_confidence_interval(model1_scores)
    m2_mean, m2_lower, m2_upper = bootstrap_confidence_interval(model2_scores)
    
    return {
        'metric': metric_name,
        'n_samples': len(common_files),
        'model1': {
            'mean': m1_mean,
            'ci_95': (m1_lower, m1_upper),
            'scores': model1_scores
        },
        'model2': {
            'mean': m2_mean,
            'ci_95': (m2_lower, m2_upper),
            'scores': model2_scores
        },
        'statistical_test': test_result,
        'winner': _determine_winner(m1_mean, m2_mean, test_result, metric_name)
    }


def batch_compare_models(
    results_dict: Dict[str, Dict[str, Any]],
    metric_name: str = 'weighted_score',
    use_parametric: bool = True
) -> List[Dict[str, Any]]:
    """
    Perform pairwise comparisons between all models.
    
    Args:
        results_dict: Dictionary mapping model_id to results
        metric_name: Metric to compare
        use_parametric: Whether to use parametric tests
        
    Returns:
        List of comparison results for all model pairs
    """
    model_ids = sorted(results_dict.keys())
    comparisons = []
    
    for i, model1_id in enumerate(model_ids):
        for model2_id in model_ids[i+1:]:
            comparison = compare_models(
                results_dict[model1_id],
                results_dict[model2_id],
                metric_name,
                use_parametric
            )
            comparison['model1_id'] = model1_id
            comparison['model2_id'] = model2_id
            comparisons.append(comparison)
    
    return comparisons


def _interpret_significance(p_value: float, is_significant: bool) -> str:
    """Generate human-readable interpretation of significance test."""
    if is_significant:
        if p_value < 0.001:
            return "Highly significant difference (p < 0.001)"
        elif p_value < 0.01:
            return "Very significant difference (p < 0.01)"
        else:
            return "Significant difference (p < 0.05)"
    else:
        return "No significant difference (p >= 0.05)"


def _determine_winner(mean1: float, mean2: float, test_result: Dict, metric_name: str) -> str:
    """
    Determine which model performs better based on statistical test.
    
    For error metrics (CER, WER, NED), lower is better.
    For other metrics, higher is better.
    """
    error_metrics = {'cer', 'wer', 'ned'}
    lower_is_better = metric_name.lower() in error_metrics
    
    if not test_result.get('significant', False):
        return "No significant difference"
    
    if lower_is_better:
        return "Model 1" if mean1 < mean2 else "Model 2"
    else:
        return "Model 1" if mean1 > mean2 else "Model 2"


def calculate_cohens_d(group1: List[float], group2: List[float]) -> float:
    """
    Calculate Cohen's d effect size for two independent groups.
    
    Args:
        group1: Scores from first group
        group2: Scores from second group
        
    Returns:
        Cohen's d value (small: 0.2, medium: 0.5, large: 0.8)
    """
    if not group1 or not group2:
        return 0.0
    
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def calculate_agreement_kappa(
    annotator1_labels: List[Any],
    annotator2_labels: List[Any]
) -> Dict[str, float]:
    """
    Calculate Cohen's Kappa for inter-rater agreement.
    Useful when multiple annotators label the ground truth.
    
    Args:
        annotator1_labels: Labels from first annotator
        annotator2_labels: Labels from second annotator
        
    Returns:
        Dictionary with kappa score and interpretation
    """
    if len(annotator1_labels) != len(annotator2_labels):
        raise ValueError("Both annotators must label the same number of samples")
    
    # Create contingency matrix
    unique_labels = sorted(set(annotator1_labels) | set(annotator2_labels))
    n_labels = len(unique_labels)
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    
    confusion = np.zeros((n_labels, n_labels))
    for l1, l2 in zip(annotator1_labels, annotator2_labels):
        confusion[label_to_idx[l1], label_to_idx[l2]] += 1
    
    # Calculate observed agreement
    n_total = len(annotator1_labels)
    observed_agreement = np.trace(confusion) / n_total
    
    # Calculate expected agreement
    row_marginals = confusion.sum(axis=1) / n_total
    col_marginals = confusion.sum(axis=0) / n_total
    expected_agreement = np.sum(row_marginals * col_marginals)
    
    # Calculate Cohen's Kappa
    if expected_agreement == 1:
        kappa = 1.0
    else:
        kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
    
    # Interpret kappa
    if kappa < 0:
        interpretation = "Poor agreement (worse than chance)"
    elif kappa < 0.20:
        interpretation = "Slight agreement"
    elif kappa < 0.40:
        interpretation = "Fair agreement"
    elif kappa < 0.60:
        interpretation = "Moderate agreement"
    elif kappa < 0.80:
        interpretation = "Substantial agreement"
    else:
        interpretation = "Almost perfect agreement"
    
    return {
        'kappa': float(kappa),
        'observed_agreement': float(observed_agreement),
        'expected_agreement': float(expected_agreement),
        'interpretation': interpretation
    }
