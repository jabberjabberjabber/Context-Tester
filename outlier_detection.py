import statistics
from typing import Dict, List, Set, Tuple, Any

def detect_outliers_iqr(values: List[float], multiplier: float = 1.5) -> Tuple[Set[int], Dict[str, float]]:
    """ Detect outliers using the IQR method
    
    Args:
        values: List of numerical values to analyze
        multiplier: IQR multiplier for outlier bounds (default: 1.5)
        
    Returns:
        Tuple of (outlier_indices, iqr_stats)
    """
    if len(values) < 4:  # Need at least 4 values for meaningful IQR
        return set(), {}
    
    sorted_values = sorted(values)
    n = len(sorted_values)
    
    # Calculate quartiles
    q1_idx = (n - 1) * 0.25
    q3_idx = (n - 1) * 0.75
    
    # Interpolate quartiles if needed
    if q1_idx.is_integer():
        q1 = sorted_values[int(q1_idx)]
    else:
        lower = int(q1_idx)
        upper = lower + 1
        weight = q1_idx - lower
        q1 = sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight
    
    if q3_idx.is_integer():
        q3 = sorted_values[int(q3_idx)]
    else:
        lower = int(q3_idx)
        upper = lower + 1
        weight = q3_idx - lower
        q3 = sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight
    
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    # Find outlier indices in original order
    outlier_indices = set()
    for i, value in enumerate(values):
        if value < lower_bound or value > upper_bound:
            outlier_indices.add(i)
    
    iqr_stats = {
        'q1': q1,
        'q3': q3,
        'iqr': iqr,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'median': statistics.median(values)
    }
    
    return outlier_indices, iqr_stats

def analyze_round_outliers(round_results: List[Dict], metrics: List[str] = None,
                          iqr_multiplier: float = 1.5) -> Dict[str, Any]:
    """ Analyze outliers across multiple metrics for round results
    
    Args:
        round_results: List of round result dictionaries
        metrics: List of metric names to check (default: key readability metrics)
        iqr_multiplier: IQR multiplier for outlier detection
        
    Returns:
        Dictionary containing outlier analysis results
    """
    if not round_results or len(round_results) < 4:
        return {'has_outliers': False, 'reason': 'insufficient_data'}
    
    if metrics is None:
        metrics = [
            'cloze_score',
            'vocabulary_diversity', 
            'sentence_length_variance',
            'pct_unfamiliar_words',
            'avg_sentence_length'
        ]
    
    # Filter metrics that exist in the data
    available_metrics = [m for m in metrics if m in round_results[0]]
    
    outlier_analysis = {
        'has_outliers': False,
        'total_rounds': len(round_results),
        'metrics_analyzed': available_metrics,
        'outlier_details': {},
        'outlier_rounds': set(),  # Rounds that are outliers in any metric
        'severe_outlier_rounds': set(),  # Rounds that are outliers in multiple metrics
    }
    
    for metric in available_metrics:
        values = [result[metric] for result in round_results]
        outlier_indices, iqr_stats = detect_outliers_iqr(values, iqr_multiplier)
        
        if outlier_indices:
            outlier_analysis['has_outliers'] = True
            outlier_analysis['outlier_rounds'].update(outlier_indices)
            
            outlier_analysis['outlier_details'][metric] = {
                'outlier_indices': list(outlier_indices),
                'outlier_values': [values[i] for i in outlier_indices],
                'iqr_stats': iqr_stats,
                'outlier_count': len(outlier_indices)
            }
    
    # Identify severe outliers (outliers in multiple metrics)
    round_outlier_counts = {}
    for metric_outliers in outlier_analysis['outlier_details'].values():
        for idx in metric_outliers['outlier_indices']:
            round_outlier_counts[idx] = round_outlier_counts.get(idx, 0) + 1
    
    # Rounds that are outliers in 2+ metrics are considered severe
    outlier_analysis['severe_outlier_rounds'] = {
        idx for idx, count in round_outlier_counts.items() if count >= 2
    }
    
    return outlier_analysis

def filter_outliers_from_results(round_results: List[Dict], 
                                outlier_analysis: Dict[str, Any],
                                exclude_severe_only: bool = True) -> List[Dict]:
    """ Filter outlier rounds from results
    
    Args:
        round_results: Original round results
        outlier_analysis: Output from analyze_round_outliers
        exclude_severe_only: If True, only exclude severe outliers; if False, exclude all outliers
        
    Returns:
        Filtered list of round results
    """
    if not outlier_analysis.get('has_outliers', False):
        return round_results
    
    if exclude_severe_only:
        exclude_indices = outlier_analysis['severe_outlier_rounds']
    else:
        exclude_indices = outlier_analysis['outlier_rounds']
    
    if not exclude_indices:
        return round_results
    
    filtered_results = [
        result for i, result in enumerate(round_results) 
        if i not in exclude_indices
    ]
    
    return filtered_results

def print_outlier_summary(outlier_analysis: Dict[str, Any], context_length: int):
    """ Print a summary of outlier detection results """
    if not outlier_analysis.get('has_outliers', False):
        return
    
    print(f"    ⚠️  Outliers detected in {len(outlier_analysis['outlier_details'])} metrics")
    
    for metric, details in outlier_analysis['outlier_details'].items():
        outlier_indices = details['outlier_indices']
        outlier_values = details['outlier_values']
        iqr_stats = details['iqr_stats']
        
        print(f"      {metric}:")
        print(f"        Rounds: {[i+1 for i in outlier_indices]} "
              f"(values: {[round(v, 3) for v in outlier_values]})")
        print(f"        Normal range: {iqr_stats['lower_bound']:.3f} - {iqr_stats['upper_bound']:.3f}")
    
    if outlier_analysis['severe_outlier_rounds']:
        severe_rounds = [i+1 for i in outlier_analysis['severe_outlier_rounds']]
        print(f"    🚨 Severe outliers (multiple metrics): Rounds {severe_rounds}")
