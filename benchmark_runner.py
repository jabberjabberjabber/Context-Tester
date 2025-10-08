#!/usr/bin/env python3
"""
Benchmark execution and testing logic for Context Tester.

Handles test runs, retries, and outlier detection.
"""

from typing import List, Dict, Any
from datetime import datetime
from readability_tests import analyze_text_comprehensive
from outlier_detection import analyze_round_outliers


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
        if retries >= max_retries:
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
                top_p=generation_params['top_p']
            )
        except Exception as e:
            print(f"WARNING: Generation failed for round {round_num}: {e}")
            retries += 1
            continue

        if not continuation:
            print(f"WARNING: No continuation generated for round {round_num}")
            retries += 1
            continue

        round_end_time = datetime.now()
        generation_time = (round_end_time - round_start_time).total_seconds()

        # Validate token count
        token_count = client.count_tokens(continuation)
        min_tokens = generation_params['max_tokens'] * 0.6

        if token_count < min_tokens and not ignore_min_tokens:
            print(f"Not enough tokens generated for round {round_num} ({token_count} < {min_tokens})")
            retries += 1
            continue

        # Analyze continuation
        try:
            analysis = analyze_text_comprehensive(continuation, client)
        except Exception as e:
            print(f"WARNING: Analysis failed for round {round_num}: {e}")
            retries += 1
            continue

        if not analysis:
            print(f"WARNING: Analysis returned no results for round {round_num}")
            retries += 1
            continue

        # Store successful result
        result = {
            'round_number': round_num,
            'continuation_length': len(continuation),
            'continuation_tokens': token_count,
            'generation_time_seconds': generation_time,
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
        threshold: Z-score threshold for outlier detection

    Returns:
        List of indices of outlier rounds (empty if none)
    """
    if len(results) < 3:  # Need minimum data for outlier detection
        return []

    try:
        outlier_analysis = analyze_round_outliers(results)

        if not outlier_analysis.get('has_outliers', False):
            return []

        # Extract outlier indices for severe outliers only
        outlier_indices = []

        for metric, outliers in outlier_analysis.get('outliers_by_metric', {}).items():
            for outlier_info in outliers:
                if outlier_info.get('severity') == 'severe':
                    round_idx = outlier_info.get('round_index')
                    if round_idx is not None and round_idx not in outlier_indices:
                        outlier_indices.append(round_idx)

        return sorted(outlier_indices)

    except Exception as e:
        print(f"WARNING: Outlier detection failed: {e}")
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
    from file_operations import save_context_results, average_results

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
            # Have outliers - retry those specific rounds
            num_outliers = len(outlier_indices)
            print(f"Outliers detected: {num_outliers}")

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
                    results[idx] = retry_results[i]

            # Check if outliers still present
            if check_for_outliers(results):
                print(f"Outliers still present after retry. Context {context_length} may be problematic.")

            retry_count += 1

    print(f"Max retries exceeded for context {context_length:,}")
    return False
