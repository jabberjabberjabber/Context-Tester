#!/usr/bin/env python3
"""
Benchmark execution and testing logic for Context Tester.

Handles test runs, retries, and outlier detection.
"""

from typing import List, Dict, Any
from datetime import datetime
from src.readability_tests import analyze_text_comprehensive
from src.outlier_detection import analyze_round_outliers


def run_benchmarks(
    context: str,
    rounds: int,
    client: Any,
    generation_params: dict,
    max_retries: int,
    ignore_min_tokens: bool
) -> List[dict]:
    """
    Run multiple benchmark rounds for a single context.

    Args:
        context: Context text to use
        rounds: Number of rounds to run
        client: API client for generation
        generation_params: Generation parameters (max_tokens, temperature, etc.)
        max_retries: Number of retries in case of outlier or bad round
        ignore_min_tokens: If True, don't retry on insufficient tokens

    Returns:
        List of individual round results
    """
    round_results = []
    retries = 0
    round_num = 1

    while round_num <= rounds:
        if retries > max_retries:
            print(f"Maximum number of retries reached ({retries}) for this tier")
            return []

        if rounds > 1:
            print(f"  Round {round_num}/{rounds}:")
        else:
            print(f"Generating continuation...")

        round_start_time = datetime.now()
        
        # Generate continuation
        try:
            continuation = client.generate_continuation(
                context,
                max_tokens=generation_params['max_tokens'],
                temperature=generation_params['temperature'],
                top_k=generation_params['top_k'],
                top_p=generation_params['top_p'],
                no_think=generation_params['no_think']
            )
        except Exception as e:
            print(f"WARNING: Generation failed for round {round_num}: {e}")
            retries += 1
            return False
            continue

        if not continuation:
            print(f"WARNING: No continuation generated for round {round_num}")
            retries += 1
            return False
            continue

        round_end_time = datetime.now()
        generation_time = (round_end_time - round_start_time).total_seconds()
        generation_time_ms = int(generation_time * 1000)

        # Validate token count
        token_count = client.count_tokens(continuation)
        min_tokens = generation_params['max_tokens'] * 0.6

        if token_count < min_tokens and not ignore_min_tokens:
            print(f"Not enough tokens generated for round {round_num} ({token_count} < {min_tokens})")
            retries += 1
            return False
            continue

        # Analyze continuation
        try:
            analysis = analyze_text_comprehensive(continuation, client)
        except Exception as e:
            print(f"WARNING: Analysis failed for round {round_num}: {e}")
            retries += 1
            return False
            continue

        if not analysis:
            print(f"WARNING: Analysis returned no results for round {round_num}")
            retries += 1
            return False
            continue

        # Store successful result
        result = {
            'round_number': round_num,
            'continuation_length': len(continuation),
            'continuation_tokens': token_count,
            'generation_time_seconds': generation_time,
            'generation_time_ms': generation_time_ms,
            'timestamp': round_end_time.isoformat(),
            'continuation_text': continuation,
            **analysis
        }

        round_results.append(result)
        round_num += 1

    return round_results


def check_for_outliers(results: List[dict], threshold: float = 2.0) -> List[int]:
    """
    Check for outlier rounds using statistical analysis.

    Args:
        results: List of round results
        threshold: Z-score threshold for outlier detection (deprecated, uses IQR now)

    Returns:
        List of indices of outlier rounds (empty if none)
    """
    if len(results) < 4:  # Need minimum 4 values for IQR outlier detection
        return []

    try:
        # Run IQR-based outlier analysis with multiplier=1.5 (standard)
        outlier_analysis = analyze_round_outliers(results, iqr_multiplier=1.5)

        if not outlier_analysis.get('has_outliers', False):
            return []

        # For critical metrics like cloze_score, detect ANY outlier (not just severe)
        # This is more conservative and catches degradation better
        critical_metrics = ['cloze_score', 'vocabulary_diversity']
        outlier_indices = set()

        # Check critical metrics - any outlier is significant
        for metric in critical_metrics:
            if metric in outlier_analysis.get('outlier_details', {}):
                details = outlier_analysis['outlier_details'][metric]
                indices = details.get('outlier_indices', [])
                outlier_indices.update(indices)

                # Print warning for critical metric outliers
                if indices:
                    values = details.get('outlier_values', [])
                    bounds = details.get('iqr_stats', {})

                    # Get all values for this metric to show context
                    all_values = [r.get(metric, 0) for r in results]

                    print(f"    ⚠️  Outlier detected in {metric}:")
                    print(f"        Round(s): {[i+1 for i in indices]}")
                    print(f"        Value(s): {[f'{v:.4f}' for v in values]}")
                    print(f"        All values: {[f'{v:.4f}' for v in all_values]}")
                    print(f"        Normal range: {bounds.get('lower_bound', 0):.4f} - {bounds.get('upper_bound', 0):.4f}")
                    print(f"        Median: {bounds.get('median', 0):.4f}, IQR: {bounds.get('iqr', 0):.4f}")

        # Also add severe outliers (outliers in multiple metrics)
        severe_indices = outlier_analysis.get('severe_outlier_rounds', set())
        if severe_indices:
            print(f"    🚨 Severe outliers (multiple metrics): Round(s) {[i+1 for i in severe_indices]}")
            outlier_indices.update(severe_indices)

        return sorted(list(outlier_indices))

    except Exception as e:
        print(f"WARNING: Outlier detection failed: {e}")
        import traceback
        traceback.print_exc()
        return []


def run_context_test_with_retries(
    context: str,
    context_length: int,
    context_info: dict,
    rounds: int,
    client: Any,
    generation_params: dict,
    max_retries: int,
    ignore_min_tokens: bool,
    output_dir: Any
) -> bool:
    """
    Run benchmark for a single context length with retry logic.

    Args:
        context: Context text
        context_length: Target context length
        context_info: Context metadata
        rounds: Number of rounds per context
        client: API client
        generation_params: Generation parameters
        max_retries: Maximum retry attempts
        ignore_min_tokens: Skip minimum token validation
        output_dir: Output directory for results

    Returns:
        True if successful, False otherwise
    """
    from src.file_operations import save_context_results, average_results

    retry_count = 0

    while retry_count <= max_retries:
        if retry_count > 0:
            print(f"\nRetry {retry_count}/{max_retries} for context {context_length:,}")

        results = run_benchmarks(
            context,
            rounds,
            client,
            generation_params,
            max_retries,
            ignore_min_tokens
        )

        if not results:
            print(f"WARNING: No successful rounds for context length {context_length}")
            retry_count += 1
            if retry_count > max_retries:
                return False
            continue

        # Check for outliers
        outlier_indices = check_for_outliers(results)

        if not outlier_indices and len(results) >= rounds:
            # Success - no outliers and all rounds complete
            save_context_results(
                output_dir,
                context_length,
                context_info,
                results,
                generation_params
            )

            # Print summary
            averaged = average_results(results)
            print(f"\nAveraged Results (n={len(results)}):")
            print(f"  Cloze Score: {averaged['cloze_score']:6.2f}")
            if 'cloze_score_std' in averaged:
                print(f"  (±{averaged['cloze_score_std']:4.2f})")
            print(f"  Reading Level: {averaged['reading_level']:>6}")
            print(f"  Vocabulary Diversity: {averaged['vocabulary_diversity']:5.3f}")
            if 'vocab_diversity_std' in averaged:
                print(f"  (±{averaged['vocab_diversity_std']:5.3f})")

            return True

        else:
            # Have outliers - but check if too many to be valid
            num_outliers = len(outlier_indices)
            outlier_percentage = (num_outliers / len(results)) * 100

            # If more than 30% are outliers, the distribution itself is problematic
            if outlier_percentage > 30:
                print(f"\n⚠️  {num_outliers}/{len(results)} rounds ({outlier_percentage:.0f}%) flagged as outliers")
                print(f"      This suggests high variance or systematic issues, not true outliers")
                print(f"      Saving results as-is (variance may indicate real degradation)")
                save_context_results(
                    output_dir,
                    context_length,
                    context_info,
                    results,
                    generation_params
                )

                averaged = average_results(results)
                print(f"\nAveraged Results (n={len(results)}):")
                print(f"  Cloze Score: {averaged['cloze_score']:6.2f} (±{averaged.get('cloze_score_std', 0):4.2f})")
                print(f"  Vocabulary Diversity: {averaged['vocabulary_diversity']:5.3f} (±{averaged.get('vocab_diversity_std', 0):5.3f})")

                return True

            # Try replacing outliers ONCE
            print(f"\n⚠️  {num_outliers} outlier round(s) detected, replacing once...")

            retry_results = run_benchmarks(
                context,
                num_outliers,
                client,
                generation_params,
                1,  # Single retry for outlier replacement
                ignore_min_tokens
            )

            # Replace outliers with new results
            if len(retry_results) == num_outliers:
                for i, idx in enumerate(outlier_indices):
                    print(f"      Replacing round {idx+1} with new generation")
                    results[idx] = retry_results[i]
                    # Update round number to indicate it's a replacement
                    results[idx]['round_number'] = idx + 1
            else:
                print(f"      WARNING: Only got {len(retry_results)} replacements for {num_outliers} outliers")

            # Save results regardless of whether new outliers appear
            # Reasoning: One replacement attempt is enough - if issues persist, they may be real
            print(f"      Saving results (with {num_outliers} round(s) replaced)")

            save_context_results(
                output_dir,
                context_length,
                context_info,
                results,
                generation_params
            )

            # Print summary
            averaged = average_results(results)
            print(f"\nAveraged Results (n={len(results)}):")
            print(f"  Cloze Score: {averaged['cloze_score']:6.2f}")
            if 'cloze_score_std' in averaged:
                print(f"  (±{averaged['cloze_score_std']:4.2f})")
            print(f"  Reading Level: {averaged['reading_level']:>6}")
            print(f"  Vocabulary Diversity: {averaged['vocabulary_diversity']:5.3f}")
            if 'vocab_diversity_std' in averaged:
                print(f"  (±{averaged['vocab_diversity_std']:5.3f})")

            return True

    # Should not reach here, but just in case
    print(f"Max retries exceeded for context {context_length:,}")
    return False
