#!/usr/bin/env python3
"""
Statistical analysis functions for Context Tester.

Provides advanced statistical analysis including:
- Composite degradation scores
- Baseline normalization
- Bootstrap confidence intervals
- Significance testing
- Effect size calculations
- Trend detection
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy import stats
from scipy.stats import bootstrap


def calculate_composite_score(
    vocab_diversity: float,
    cloze_score: float,
    adjacent_coherence: float,
    bigram_repetition: float
) -> float:
    """Calculate composite degradation score from four metrics.

    Higher score = more degradation (worse performance)

    Args:
        vocab_diversity: Vocabulary diversity (lower = worse)
        cloze_score: Cloze readability score (higher = worse)
        adjacent_coherence: Adjacent sentence similarity (lower = worse)
        bigram_repetition: Bigram repetition rate (higher = worse)

    Returns:
        Composite score (0-100 scale, higher = more degradation)
    """
    # Normalize each metric to 0-1 range based on typical values
    # Vocab diversity: typical range 0.3-0.6, lower is worse
    vocab_norm = 1.0 - np.clip((vocab_diversity - 0.3) / 0.3, 0, 1)

    # Cloze score: typical range 15-40, higher is worse
    cloze_norm = np.clip((cloze_score - 15) / 25, 0, 1)

    # Adjacent coherence: typical range 0.3-0.7, lower is worse
    coherence_norm = 1.0 - np.clip((adjacent_coherence - 0.3) / 0.4, 0, 1)

    # Bigram repetition: typical range 0.0-0.3, higher is worse
    repetition_norm = np.clip(bigram_repetition / 0.3, 0, 1)

    # Weighted average (equal weights for now)
    composite = (vocab_norm + cloze_norm + coherence_norm + repetition_norm) / 4.0

    # Scale to 0-100
    return composite * 100


def normalize_to_baseline(
    values: np.ndarray,
    baseline_mean: float,
    baseline_std: float,
    method: str = 'zscore'
) -> np.ndarray:
    """Normalize values relative to baseline.

    Args:
        values: Array of values to normalize
        baseline_mean: Mean of baseline (smallest context) values
        baseline_std: Standard deviation of baseline values
        method: 'zscore' or 'percent'

    Returns:
        Normalized values
    """
    if method == 'zscore':
        if baseline_std == 0:
            return np.zeros_like(values)
        return (values - baseline_mean) / baseline_std
    elif method == 'percent':
        if baseline_mean == 0:
            return np.zeros_like(values)
        return ((values - baseline_mean) / baseline_mean) * 100
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def bootstrap_confidence_interval(
    data: np.ndarray,
    confidence_level: float = 0.95,
    n_resamples: int = 10000,
    statistic = np.mean
) -> Tuple[float, float, float]:
    """Calculate bootstrap confidence interval.

    Args:
        data: Array of observed values
        confidence_level: Confidence level (default: 0.95)
        n_resamples: Number of bootstrap resamples
        statistic: Function to compute statistic (default: mean)

    Returns:
        Tuple of (point_estimate, lower_ci, upper_ci)
    """
    if len(data) < 2:
        point = statistic(data) if len(data) == 1 else 0.0
        return (point, point, point)

    # Calculate point estimate
    point_estimate = statistic(data)

    # Bootstrap CI
    res = bootstrap(
        (data,),
        statistic,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method='percentile',
        random_state=42
    )

    return (point_estimate, res.confidence_interval.low, res.confidence_interval.high)


def mann_whitney_test(
    baseline_samples: np.ndarray,
    test_samples: np.ndarray,
    alternative: str = 'two-sided'
) -> Tuple[float, float]:
    """Perform Mann-Whitney U test.

    Args:
        baseline_samples: Baseline (control) samples
        test_samples: Test samples to compare
        alternative: 'two-sided', 'less', or 'greater'

    Returns:
        Tuple of (u_statistic, p_value)
    """
    if len(baseline_samples) < 2 or len(test_samples) < 2:
        return (0.0, 1.0)

    u_stat, p_val = stats.mannwhitneyu(
        baseline_samples,
        test_samples,
        alternative=alternative
    )

    return (float(u_stat), float(p_val))


def calculate_effect_size(
    baseline_samples: np.ndarray,
    test_samples: np.ndarray
) -> float:
    """Calculate rank-biserial correlation (effect size for Mann-Whitney U).

    Args:
        baseline_samples: Baseline (control) samples
        test_samples: Test samples to compare

    Returns:
        Effect size r (-1 to 1)
        Interpretation: |r| < 0.3 = small, 0.3-0.5 = medium, > 0.5 = large
    """
    if len(baseline_samples) < 1 or len(test_samples) < 1:
        return 0.0

    n1 = len(baseline_samples)
    n2 = len(test_samples)

    u_stat, _ = stats.mannwhitneyu(baseline_samples, test_samples)

    # Rank-biserial correlation
    r = 1 - (2 * u_stat) / (n1 * n2)

    return float(r)


def spearman_correlation(
    x: np.ndarray,
    y: np.ndarray
) -> Tuple[float, float]:
    """Calculate Spearman rank correlation.

    Args:
        x: Independent variable (e.g., context sizes)
        y: Dependent variable (e.g., metric values)

    Returns:
        Tuple of (rho, p_value)
        Interpretation: |rho| > 0.7 and p < 0.05 = strong trend
    """
    if len(x) < 3 or len(y) < 3:
        return (0.0, 1.0)

    rho, p_val = stats.spearmanr(x, y)

    return (float(rho), float(p_val))


def multiple_comparison_correction(
    p_values: List[float],
    method: str = 'holm',
    alpha: float = 0.05
) -> Tuple[List[bool], List[float]]:
    """Apply multiple comparison correction.

    Args:
        p_values: List of p-values from multiple tests
        method: Correction method ('holm', 'bonferroni', 'fdr_bh')
        alpha: Significance level

    Returns:
        Tuple of (reject_list, corrected_p_values)
    """
    from statsmodels.stats.multitest import multipletests

    if not p_values:
        return ([], [])

    reject, corrected_p, _, _ = multipletests(
        p_values,
        method=method,
        alpha=alpha
    )

    return (reject.tolist(), corrected_p.tolist())


def analyze_degradation_pattern(
    context_sizes: List[int],
    metric_values: List[float],
    baseline_samples: List[float]
) -> Dict:
    """Comprehensive degradation analysis for a single metric.

    Args:
        context_sizes: List of context sizes tested
        metric_values: Mean metric value at each context size
        baseline_samples: Individual samples from baseline (smallest) context

    Returns:
        Dictionary with analysis results
    """
    # Convert to numpy arrays
    contexts = np.array(context_sizes)
    values = np.array(metric_values)
    baseline = np.array(baseline_samples)

    baseline_mean = np.mean(baseline)
    baseline_std = np.std(baseline)

    # Trend detection
    rho, trend_p = spearman_correlation(contexts, values)

    # Baseline normalization
    z_scores = normalize_to_baseline(values, baseline_mean, baseline_std, 'zscore')
    pct_changes = normalize_to_baseline(values, baseline_mean, baseline_std, 'percent')

    return {
        'baseline_mean': float(baseline_mean),
        'baseline_std': float(baseline_std),
        'trend_correlation': float(rho),
        'trend_p_value': float(trend_p),
        'has_significant_trend': abs(rho) > 0.7 and trend_p < 0.05,
        'z_scores': z_scores.tolist(),
        'percent_changes': pct_changes.tolist(),
        'max_deviation_zscore': float(np.max(np.abs(z_scores))),
        'max_deviation_percent': float(np.max(np.abs(pct_changes)))
    }


def calculate_composite_scores_for_dataset(
    results_by_context: Dict[int, List[Dict]]
) -> Dict[int, Dict]:
    """Calculate composite scores for all context sizes.

    Args:
        results_by_context: Dict mapping context_size -> list of round results

    Returns:
        Dict mapping context_size -> {
            'composite_scores': list of composite scores for each round,
            'mean': mean composite score,
            'ci_lower': lower confidence interval,
            'ci_upper': upper confidence interval
        }
    """
    composite_by_context = {}

    for context_size, rounds in results_by_context.items():
        composite_scores = []

        for round_data in rounds:
            score = calculate_composite_score(
                round_data.get('vocabulary_diversity', 0.5),
                round_data.get('cloze_score', 25.0),
                round_data.get('adjacent_coherence', 0.5),
                round_data.get('bigram_repetition_rate', 0.1)
            )
            composite_scores.append(score)

        # Calculate bootstrap CI
        scores_array = np.array(composite_scores)
        mean, ci_low, ci_high = bootstrap_confidence_interval(scores_array)

        composite_by_context[context_size] = {
            'composite_scores': composite_scores,
            'mean': mean,
            'ci_lower': ci_low,
            'ci_upper': ci_high,
            'std': float(np.std(scores_array))
        }

    return composite_by_context


def comprehensive_statistical_analysis(
    results_by_context: Dict[int, List[Dict]]
) -> Dict:
    """Perform comprehensive statistical analysis on context test results.

    Args:
        results_by_context: Dict mapping context_size -> list of round results

    Returns:
        Dictionary with comprehensive analysis including:
        - Composite scores with CI
        - Baseline-normalized metrics
        - Significance tests
        - Effect sizes
        - Trend analysis
    """
    if not results_by_context:
        return {}

    # Get sorted context sizes
    context_sizes = sorted(results_by_context.keys())
    baseline_context = context_sizes[0]

    # Calculate composite scores
    composite_analysis = calculate_composite_scores_for_dataset(results_by_context)

    # Extract composite means for trend analysis
    composite_means = [composite_analysis[c]['mean'] for c in context_sizes]
    baseline_composite = composite_analysis[baseline_context]['composite_scores']

    # Trend analysis on composite score
    rho, trend_p = spearman_correlation(
        np.array(context_sizes),
        np.array(composite_means)
    )

    # Significance testing for each context size vs baseline
    significance_tests = {}
    effect_sizes = {}

    for context_size in context_sizes[1:]:  # Skip baseline
        test_composites = composite_analysis[context_size]['composite_scores']

        u_stat, p_val = mann_whitney_test(
            np.array(baseline_composite),
            np.array(test_composites)
        )

        effect_size = calculate_effect_size(
            np.array(baseline_composite),
            np.array(test_composites)
        )

        significance_tests[context_size] = {
            'u_statistic': u_stat,
            'p_value': p_val,
            'significant': p_val < 0.05
        }

        effect_sizes[context_size] = {
            'effect_size': effect_size,
            'magnitude': (
                'large' if abs(effect_size) > 0.5 else
                'medium' if abs(effect_size) > 0.3 else
                'small'
            )
        }

    # Multiple comparison correction
    p_values = [significance_tests[c]['p_value'] for c in context_sizes[1:]]
    reject_list, corrected_p = multiple_comparison_correction(p_values)

    for i, context_size in enumerate(context_sizes[1:]):
        significance_tests[context_size]['corrected_p_value'] = corrected_p[i]
        significance_tests[context_size]['significant_corrected'] = reject_list[i]

    return {
        'context_sizes': context_sizes,
        'baseline_context': baseline_context,
        'composite_scores': composite_analysis,
        'trend_correlation': rho,
        'trend_p_value': trend_p,
        'has_degradation_trend': rho > 0.5 and trend_p < 0.05,
        'significance_tests': significance_tests,
        'effect_sizes': effect_sizes,
        'summary': {
            'degradation_detected': rho > 0.5 and trend_p < 0.05,
            'max_effect_size': max([abs(e['effect_size']) for e in effect_sizes.values()]) if effect_sizes else 0.0,
            'num_significant_degradations': sum([1 for t in significance_tests.values() if t['significant_corrected']])
        }
    }
