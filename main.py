#!/usr/bin/env python3
"""
Model Readability Degradation Test - Functional Architecture

Tests LLM output quality degradation at increasing context lengths by analyzing
the readability complexity of generated text continuations using Cloze scores.
Saves results incrementally and retries outliers.

Based on observations that models exhibit "flattening" patterns around 8k+ context,
where sentence structure becomes repetitive and vocabulary diversity drops.
"""

import argparse
import json
import math
import re
import warnings
import csv
import statistics
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Literal, TypeAlias
import os
import shutil

import requests
from readability_tests import (
    initialize_word_list, 
    analyze_text_comprehensive, 
    analyze_text_readability,
    reading_level_from_cloze,
    is_power_of_two
)

from generate_plot import make_png

from find_last_sentence import find_last_sentence_ending

from requests.exceptions import RequestException
from bs4 import BeautifulSoup
from extractous import Extractor
from streaming_api import StreamingAPIClient

from outlier_detection import (
    analyze_round_outliers, 
    filter_outliers_from_results,
    print_outlier_summary
)

warnings.filterwarnings("ignore", category=UserWarning, module="bs4")

# ================================
# TEXT PROCESSING FUNCTIONS
# ================================

def normalize_content(content: str) -> str:
    """ Convert fixed-width text and normalize characters """
    content = unicodedata.normalize('NFKC', content)
    content = content.replace('--', '—')
    
    # Normalize quotes to straight quotes (optional)
    content = content.replace('"', '"').replace('"', '"')
    content = content.replace(''', "'").replace(''', "'")
    
    text = content.replace('\r\n', '\n').replace('\r', '\n')

    paragraphs = text.split('\n\n')
    
    # Fix windows line breaks and convert fixed width text to wrap
    result = '\n\n'.join(para.replace('\n', ' ') for para in paragraphs)
    result = result.replace('\n\n', '\n\n    ')
    return result

def load_reference_text(file_path: str) -> str:
    """ Load reference text from file """
    extractor = Extractor()
    
    content, metadata = extractor.extract_file_to_string(file_path)
    print(f"Loaded reference text: {metadata.get('resourceName', file_path)}")
    print(f"Content type: {metadata.get('Content-Type', 'Unknown')}")
    
    # Debug text loading
    if not content or not content.strip():
        raise ValueError(f"File {file_path} appears to be empty or could not be read")
    content_size = int(len(content) * 0.8)
    normalized_content = normalize_content(content[:content_size])
    print(f"Text preview (first 200 chars): {normalized_content[:200]!r}")
    print(f"Total characters: {len(normalized_content):,}")
    return normalized_content

def prepare_working_tokens(text: str, max_context: int, client: StreamingAPIClient) -> List[str]:
    """ Tokenize text and prepare working token set 
    
    Args:
        text: Reference text to tokenize
        max_context: Maximum context length we'll test
        client: API client for tokenization
        
    Returns:
        List of tokens for testing
    """

    min_required = int(max_context * 1.2)
    
    # Get the working portion first
    all_tokens = client.tokenize_text_batched(text)
    
    if len(all_tokens) < min_required:
        # Handle repetition case
        repeats_needed = (min_required // len(all_tokens)) + 1
        working_tokens = (all_tokens * repeats_needed)[:min_required]
    else:
        working_tokens = all_tokens[:min_required]
    #print(f"Tokens: {working_tokens}")
    # Convert back to text and THEN prune with regex
    working_text = client.tokens_to_text_batched(working_tokens)
    pruned_text = client.prune_text(working_text, max_context, len(working_tokens))
    
    # Final tokenization
    final_tokens = client.tokenize_text_batched(pruned_text)
    return final_tokens

# ================================
# CONTEXT GENERATION FUNCTIONS
# ================================

def generate_context_lengths(max_context: int, divisions: int = 1, start_context: Optional[int] = None) -> List[int]:
    """ Generate power-of-2 context lengths to test with subdivisions
    
    Args:
        max_context: Maximum context length
        divisions: Number of splits per power of 2 tier
        start_context: Optional starting context size (skip smaller sizes)
        
    Returns:
        List of context lengths in tokens
    """
    # Generate base powers of 2: 1k, 2k, 4k, 8k, 16k, 32k, etc.
    tiers = []
    power = 10  # Start at 2^10 = 1024
    
    while True:
        length = 2 ** power
        if length > max_context:
            break
        
        tiers.append(length)
        power += 1
    
    if not tiers:
        # Fallback for very small max_context
        return [min(1024, max_context)]
    
    # Filter tiers based on start_context if specified
    if start_context is not None:
        tiers = [tier for tier in tiers if tier >= start_context]
        if not tiers:
            # If start_context is larger than all tiers, just use max_context
            return [min(start_context, max_context)]
        print(f"Starting from context size: {start_context:,} tokens")
    
    # Apply subdivision logic
    context_lengths = [tiers[0]]
    
    for i in range(len(tiers) - 1):
        start, end = tiers[i], tiers[i + 1]
        step = (end - start) / divisions
        
        # Insert divisions-1 intermediate values
        for j in range(1, divisions):
            context_lengths.append(int(start + step * j))
        
        context_lengths.append(end)
    
    print(f"Generated context lengths: {context_lengths}")
    return context_lengths

def build_context_window(working_tokens: List[str], context_length: int, 
                        client: StreamingAPIClient, max_tokens: int) -> str:
    """ Build context window from working tokens using backward expansion
    
    Args:
        working_tokens: List of tokens to work with
        context_length: Number of tokens to include in context
        client: API client for token operations
        max_tokens: Maximum tokens for generation (to adjust context)
        
    Returns:
        Context text for this tier, trimmed to natural boundaries
    """
    if context_length > len(working_tokens):
        # Use all available working tokens
        context_text = client.tokens_to_text_batched(working_tokens)
        return context_text
    
    # Adjust context length to account for generation tokens
    adjusted_length = context_length
    if context_length >= (max_tokens * 2):
        adjusted_length = context_length - max_tokens
    
    # Convert tokens to text
    target_tokens = working_tokens[-adjusted_length:]
    context_text = client.tokens_to_text_batched(target_tokens)
    
    print(f"  Context window: {adjusted_length:,} tokens")
    return context_text

# ================================
# BENCHMARK EXECUTION FUNCTIONS
# ================================

def run_benchmarks(context: str, rounds: int, client: StreamingAPIClient, 
                  generation_params: dict) -> List[dict]:
    """ Run multiple benchmark rounds for a single context
    
    Args:
        context: Context text to use
        rounds: Number of rounds to run
        client: API client for generation
        generation_params: Generation parameters (max_tokens, temperature, etc.)
        
    Returns:
        List of individual round results
    """
    round_results = []
    
    for round_num in range(rounds):
        if rounds > 1:
            print(f"  Round {round_num + 1}/{rounds}:")
        else:
            print(f"Generating continuation...")
        
        round_start_time = datetime.now()
        
        try:
            continuation = client.generate_continuation(
                context, 
                max_tokens=generation_params['max_tokens'],
                temperature=generation_params['temperature'],
                top_k=generation_params['top_k'],
                top_p=generation_params['top_p'],
                min_p=generation_params['min_p'],
                rep_pen=generation_params['rep_pen']
            )
        except Exception as e:
            print(f"WARNING: Generation failed for round {round_num + 1}: {e}")
            continue
        
        if not continuation:
            print(f"WARNING: No continuation generated for round {round_num + 1}")
            continue

        if client.count_tokens(continuation) < (generation_params['max_tokens'] * 0.8):
            print(f"Not enough tokens generated for round {round_num + 1}")
            continue
        
        round_end_time = datetime.now()
        generation_time = (round_end_time - round_start_time).total_seconds()
        
        try:
            analysis = analyze_text_comprehensive(continuation, client)
        except Exception as e:
            print(f"WARNING: Analysis failed for round {round_num + 1}: {e}")
            continue
        
        result = {
            'round_number': round_num + 1,
            'continuation_length': len(continuation),
            'continuation_tokens': client.count_tokens(continuation),
            'generation_time_seconds': generation_time,
            'timestamp': round_end_time.isoformat(),
            'continuation_text': continuation,
            **analysis
        }
        
        round_results.append(result)
        
        if rounds > 1:
            print(f"    Cloze: {analysis['cloze_score']:6.2f}, "
                  f"Level: {analysis['reading_level']:>6}, "
                  f"Vocab: {analysis['vocabulary_diversity']:5.3f}")
    
    return round_results

def check_for_outliers(results: List[dict], threshold: float = 2.0) -> List[int]:
    """ Check for outlier rounds using statistical analysis
    
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

# ================================
# STATE MANAGEMENT FUNCTIONS
# ================================

def create_output_directory(model_name: str, text_name: str, timestamp: str) -> Path:
    """ Create output directory for this experiment """
    clean_model = model_name.replace('/', '-').replace('\\', '-').replace(':', '-')
    clean_text = text_name.replace('/', '-').replace('\\', '-').replace(':', '-')
    
    dir_name = f"{clean_model}-{clean_text}-{timestamp}"
    output_dir = Path("results") / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir

def save_experiment_metadata(output_dir: Path, metadata: dict):
    """ Save experiment metadata """
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

def save_context_results(output_dir: Path, context_length: int, context_info: dict, 
                        results: List[dict], generation_params: dict):
    """ Save results for a single context length """
    filename = f"context_{context_length}_results.json"
    filepath = output_dir / filename
    
    # Calculate averaged stats
    averaged_stats = average_results(results) if results else {}
    
    data = {
        'context_length': context_length,
        'context_info': context_info,
        'generation_params': generation_params,
        'num_rounds': len(results),
        'individual_rounds': results,
        'averaged_stats': averaged_stats,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved results to: {filepath}")

def average_results(round_results: List[Dict]) -> Dict[str, Any]:
    """ Average results across multiple rounds """
    if not round_results:
        return {}
    
    if len(round_results) == 1:
        result = round_results[0].copy()
        result['num_rounds'] = 1
        result['cloze_score_std'] = 0.0
        result['vocab_diversity_std'] = 0.0
        return result
    
    numerical_fields = [
        'pct_unfamiliar_words', 'avg_sentence_length', 'cloze_score',
        'word_count', 'sentence_count', 'sentence_length_variance',
        'vocabulary_diversity', 'continuation_length',
        'bigram_repetition_rate', 'trigram_repetition_rate', 'unique_word_ratio_100',
        'word_entropy', 'char_entropy',
        'comma_density', 'semicolon_density', 'question_density', 'exclamation_density',
        'avg_syllables_per_word', 'long_word_ratio', 'function_word_ratio',
        'sentence_length_skewness', 'sentence_length_kurtosis', 'avg_word_length'
    ]
    
    averaged = {}
    
    for key, value in round_results[0].items():
        if key not in numerical_fields:
            averaged[key] = value
    
    # Average numerical fields
    for field in numerical_fields:
        values = [result[field] for result in round_results if field in result and result[field] is not None]
        if values:
            averaged[field] = round(statistics.mean(values), 4)
        else:
            averaged[field] = 0.0
    
    if 'cloze_score' in averaged:
        averaged['reading_level'] = reading_level_from_cloze(averaged['cloze_score'])
    
    averaged['num_rounds'] = len(round_results)
    
    cloze_values = [r['cloze_score'] for r in round_results if 'cloze_score' in r and r['cloze_score'] is not None]
    vocab_values = [r['vocabulary_diversity'] for r in round_results if 'vocabulary_diversity' in r and r['vocabulary_diversity'] is not None]
    
    averaged['cloze_score_std'] = round(statistics.stdev(cloze_values), 3) if len(cloze_values) > 1 else 0.0
    averaged['vocab_diversity_std'] = round(statistics.stdev(vocab_values), 4) if len(vocab_values) > 1 else 0.0
    
    return averaged

# ================================
# ANALYSIS FUNCTIONS
# ================================

def run_analysis(results_dir: Path, model_id: Optional[str] = None) -> dict:
    """ Load all saved results and perform degradation analysis 
    
    Args:
        results_dir: Directory containing saved context results
        model_id: Optional model identifier for filename generation
        
    Returns:
        Analysis results dictionary
    """
    print(f"\n{'='*60}")
    print("RUNNING DEGRADATION ANALYSIS")
    print(f"{'='*60}")
    
    # Load metadata
    metadata_file = results_dir / "metadata.json"
    if not metadata_file.exists():
        raise ValueError(f"No metadata found in {results_dir}")
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Load all context results
    context_files = list(results_dir.glob("context_*_results.json"))
    if not context_files:
        raise ValueError(f"No context results found in {results_dir}")
    
    context_results = []
    for filepath in sorted(context_files):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            context_results.append(data)
    
    # Sort by context length
    context_results.sort(key=lambda x: x['context_length'])
    
    # Extract averaged stats for analysis
    averaged_results = []
    for data in context_results:
        if data['averaged_stats']:
            stats = data['averaged_stats'].copy()
            stats['context_length'] = data['context_length']
            averaged_results.append(stats)
    
    if len(averaged_results) < 2:
        print("Insufficient data for trend analysis")
        return {'averaged_results': averaged_results}
    
    # Analyze trends
    analysis = analyze_degradation_trends(averaged_results, metadata)
    
    # Save analysis results
    analysis_file = results_dir / "degradation_analysis.json"
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    # Generate CSV for plotting
    csv_file = save_results_csv(averaged_results, results_dir, metadata, model_id)
    
    # Save generation outputs
    save_generation_outputs(results_dir, metadata, model_id)
    
    # Generate plot
    try:
        make_png(averaged_results, str(csv_file))
        print(f"Plot saved to: {csv_file.with_suffix('.png')}")
    except Exception as e:
        print(f"Plot generation failed: {e}")
    
    return analysis

def analyze_degradation_trends(results: List[dict], metadata: dict) -> dict:
    """ Analyze degradation trends across context lengths """
    print(f"\nDEGRADATION ANALYSIS")
    if metadata.get('experiment_metadata', {}).get('num_rounds', 1) > 1:
        rounds = metadata['experiment_metadata']['num_rounds']
        print(f"(Based on averages of {rounds} rounds each)")
    print("All continuations from SAME story position")
    print("-" * 50)
    
    # Calculate trends
    cloze_scores = [r['cloze_score'] for r in results]
    vocab_diversity = [r['vocabulary_diversity'] for r in results]
    sentence_variance = [r['sentence_length_variance'] for r in results]
    
    print(f"\nCloze Score Trend:")
    print(f"  Starting: {cloze_scores[0]:6.2f}")
    print(f"  Ending:   {cloze_scores[-1]:6.2f}")
    change = cloze_scores[-1] - cloze_scores[0]
    print(f"  Change:   {change:+6.2f} {'(simpler text)' if change > 0 else '(more complex text)'}")
    
    print(f"\nVocabulary Diversity Trend:")
    print(f"  Starting: {vocab_diversity[0]:6.3f}")
    print(f"  Ending:   {vocab_diversity[-1]:6.3f}")
    vocab_change = vocab_diversity[-1] - vocab_diversity[0]
    print(f"  Change:   {vocab_change:+6.3f} {'(more repetitive)' if vocab_change < 0 else '(more diverse)'}")
    
    # Look for degradation points
    degradation_points = []
    
    for i in range(1, len(results)):
        prev_score = cloze_scores[i-1]
        curr_score = cloze_scores[i]
        prev_vocab = vocab_diversity[i-1]
        curr_vocab = vocab_diversity[i]
        prev_variance = sentence_variance[i-1]
        curr_variance = sentence_variance[i]
        
        # Significant RISE in cloze score = simplification = degradation
        if curr_score - prev_score > 3.0:
            degradation_points.append({
                'context_length': results[i]['context_length'],
                'metric': 'cloze_score',
                'change': curr_score - prev_score,
                'direction': 'rose'
            })
        
        # Significant DROP in vocabulary diversity = degradation
        if prev_vocab - curr_vocab > 0.05:
            degradation_points.append({
                'context_length': results[i]['context_length'],
                'metric': 'vocabulary_diversity',
                'change': prev_vocab - curr_vocab,
                'direction': 'dropped'
            })
        
        # Significant DROP in sentence variance = degradation
        if prev_variance - curr_variance > 5.0:
            degradation_points.append({
                'context_length': results[i]['context_length'],
                'metric': 'sentence_variance',
                'change': prev_variance - curr_variance,
                'direction': 'dropped'
            })
    
    if degradation_points:
        print(f"\nDEGRADATION DETECTED:")
        for point in degradation_points:
            print(f"  At {point['context_length']:,} tokens: {point['metric']} {point['direction']} {point['change']:.3f}")
    else:
        print(f"\nNo significant degradation detected in tested range")
    
    # Print results table
    print_results_table(results)
    
    analysis = {
        'metadata': metadata,
        'results': results,
        'trends': {
            'cloze_score': {
                'start': cloze_scores[0],
                'end': cloze_scores[-1],
                'change': change
            },
            'vocabulary_diversity': {
                'start': vocab_diversity[0],
                'end': vocab_diversity[-1],
                'change': vocab_change
            }
        },
        'degradation_points': degradation_points,
        'analysis_timestamp': datetime.now().isoformat()
    }
    
    return analysis
    
def reanalyze(input_dir: Path, client: StreamingAPIClient, model_id: Optional[str] = None) -> Path:
    """ Re-analyze existing results using stored continuation texts
    
    Args:
        input_dir: Directory containing context_*_results.json files  
        client: API client for token counting and text analysis
        model_id: Optional model identifier for new result filenames
        
    Returns:
        Path to new output directory with reanalyzed results
    """
    print(f"\n{'='*60}")
    print("RE-ANALYZING EXISTING RESULTS")
    print(f"{'='*60}")
    print(f"Input directory: {input_dir}")
    
    # Load original metadata
    metadata_file = input_dir / "metadata.json"
    if not metadata_file.exists():
        raise ValueError(f"No metadata found in {input_dir}")
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        original_metadata = json.load(f)
    
    # Create new output directory for reanalyzed results
    experiment_meta = original_metadata.get('experiment_metadata', {})
    model_name = experiment_meta.get('model_name', 'unknown')
    text_name = experiment_meta.get('source_text_name', 'unknown')
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Create output directory with "reanalyzed" suffix
    clean_model = model_name.replace('/', '-').replace('\\', '-').replace(':', '-')
    clean_text = text_name.replace('/', '-').replace('\\', '-').replace(':', '-')
    dir_name = f"{clean_model}-{clean_text}-reanalyzed-{timestamp}"
    output_dir = Path("results") / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Update metadata for reanalysis
    new_metadata = original_metadata.copy()
    new_metadata['reanalysis_metadata'] = {
        'original_directory': str(input_dir),
        'reanalysis_time': datetime.now().isoformat(),
        'reanalysis_model_id': model_id,
        'analysis_method': 'reanalyzed_from_stored_continuations'
    }
    
    # Load all context result files
    context_files = list(input_dir.glob("context_*_results.json"))
    if not context_files:
        raise ValueError(f"No context results found in {input_dir}")
    
    print(f"Found {len(context_files)} context result files to reanalyze")
    
    # Process each context file
    successful_contexts = 0
    
    for context_file in sorted(context_files):
        print(f"\nReanalyzing: {context_file.name}")
        
        with open(context_file, 'r', encoding='utf-8') as f:
            context_data = json.load(f)
        
        context_length = context_data['context_length']
        context_info = context_data['context_info']
        generation_params = context_data['generation_params']
        original_rounds = context_data.get('individual_rounds', [])
        
        if not original_rounds:
            print(f"  No rounds found in {context_file.name}, skipping")
            continue
        
        print(f"  Context length: {context_length:,} tokens")
        print(f"  Reanalyzing {len(original_rounds)} rounds...")
        
        # Reanalyze each round
        reanalyzed_rounds = []
        
        for round_data in original_rounds:
            continuation_text = round_data.get('continuation_text', '')
            
            if not continuation_text:
                print(f"    Warning: No continuation text in round {round_data.get('round_number', '?')}")
                continue
            
            try:
                # Run fresh analysis on the stored continuation text
                new_analysis = analyze_text_comprehensive(continuation_text, client)
                
                # Create new round result with updated analysis
                new_round_data = {
                    'round_number': round_data.get('round_number'),
                    'continuation_length': len(continuation_text),
                    'continuation_tokens': client.count_tokens(continuation_text),
                    'generation_time_seconds': round_data.get('generation_time_seconds'),
                    'timestamp': round_data.get('timestamp'),
                    'continuation_text': continuation_text,
                    'reanalyzed_timestamp': datetime.now().isoformat(),
                    **new_analysis
                }
                
                reanalyzed_rounds.append(new_round_data)
                
            except Exception as e:
                print(f"    Warning: Failed to reanalyze round {round_data.get('round_number', '?')}: {e}")
                continue
        
        if not reanalyzed_rounds:
            print(f"  No successful reanalysis for {context_file.name}")
            continue
        
        # Save reanalyzed results for this context
        save_context_results(output_dir, context_length, context_info, reanalyzed_rounds, generation_params)
        
        # Print summary of reanalyzed results
        averaged = average_results(reanalyzed_rounds)
        print(f"  Reanalyzed Results (n={len(reanalyzed_rounds)}):")
        print(f"    Cloze Score: {averaged['cloze_score']:6.2f}")
        if 'cloze_score_std' in averaged:
            print(f"    (±{averaged['cloze_score_std']:4.2f})")
        print(f"    Reading Level: {averaged['reading_level']:>6}")
        print(f"    Vocabulary Diversity: {averaged['vocabulary_diversity']:5.3f}")
        if 'vocab_diversity_std' in averaged:
            print(f"    (±{averaged['vocab_diversity_std']:5.3f})")
        
        successful_contexts += 1
    
    # Update metadata with reanalysis completion info
    new_metadata['reanalysis_metadata']['successful_contexts'] = successful_contexts
    new_metadata['reanalysis_metadata']['total_contexts'] = len(context_files)
    
    # Save updated metadata
    save_experiment_metadata(output_dir, new_metadata)
    
    print(f"\n{'='*60}")
    print("REANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Successfully reanalyzed {successful_contexts}/{len(context_files)} context lengths")
    print(f"Results saved to: {output_dir}")
    
    # Run full analysis on reanalyzed results
    if successful_contexts >= 2:
        print(f"\nRunning degradation analysis on reanalyzed data...")
        run_analysis(output_dir, model_id)
    else:
        print(f"Insufficient reanalyzed data for trend analysis")
    
    return output_dir
    
def print_results_table(results: List[dict]):
    """ Print formatted results table """
    print(f"\nFULL RESULTS:")
    
    has_std = any('cloze_score_std' in r for r in results)
    
    if has_std:
        print(f"{'Context':>8} {'Cloze':>8} {'±Std':>6} {'Level':>8} {'Unfamiliar':>10} {'AvgSent':>8} {'Variance':>8} {'VocabDiv':>8} {'±Std':>6}")
        print("-" * 80)
        
        for result in results:
            print(f"{result['context_length']:>8,} "
                  f"{result['cloze_score']:>8.2f} "
                  f"{result.get('cloze_score_std', 0):>6.2f} "
                  f"{result['reading_level']:>8} "
                  f"{result['pct_unfamiliar_words']*100:>9.1f}% "
                  f"{result['avg_sentence_length']:>8.1f} "
                  f"{result['sentence_length_variance']:>8.1f} "
                  f"{result['vocabulary_diversity']:>8.3f} "
                  f"{result.get('vocab_diversity_std', 0):>6.3f}")
    else:
        print(f"{'Context':>8} {'Cloze':>8} {'Level':>8} {'Unfamiliar':>10} {'AvgSent':>8} {'Variance':>8} {'VocabDiv':>8}")
        print("-" * 68)
        
        for result in results:
            print(f"{result['context_length']:>8,} "
                  f"{result['cloze_score']:>8.2f} "
                  f"{result['reading_level']:>8} "
                  f"{result['pct_unfamiliar_words']*100:>9.1f}% "
                  f"{result['avg_sentence_length']:>8.1f} "
                  f"{result['sentence_length_variance']:>8.1f} "
                  f"{result['vocabulary_diversity']:>8.3f}")

def normalize_filename_part(text: str) -> str:
    """ Normalize text for safe filename usage """
    # Remove or replace problematic characters
    normalized = text.replace('/', '-').replace('\\', '-').replace(':', '-')
    normalized = re.sub(r'[<>"|?*]', '', normalized)  # Remove invalid chars
    normalized = re.sub(r'\s+', '_', normalized.strip())  # Replace spaces with underscores
    return normalized

def generate_results_filename(model_name: str, text_name: str, model_id: Optional[str] = None) -> str:
    """ Generate results filename from model name, text name, and optional model ID """
    clean_model = normalize_filename_part(model_name)
    clean_text = normalize_filename_part(text_name)
    
    if model_id and model_id.strip():
        clean_id = normalize_filename_part(model_id.strip())
        return f"{clean_model}-{clean_text}-{clean_id}"
    else:
        return f"{clean_model}-{clean_text}"

def save_results_csv(results: List[dict], output_dir: Path, metadata: dict, model_id: Optional[str] = None):
    """ Save results to CSV for plotting compatibility """
    if not results:
        return
    
    # Generate filename
    experiment_meta = metadata.get('experiment_metadata', {})
    model_name = experiment_meta.get('model_name', 'unknown')
    text_name = experiment_meta.get('source_text_name', 'unknown')
    
    filename = generate_results_filename(model_name, text_name, model_id) + ".csv"
    output_file = output_dir / filename
    
    # Add metadata to each result
    enhanced_results = []
    gen_params = experiment_meta.get('generation_params', {})
    
    for result in results:
        enhanced_result = {
            'model_name': model_name,
            'max_tokens': gen_params.get('max_tokens', 1024),
            'temperature': gen_params.get('temperature', 1.0),
            'top_k': gen_params.get('top_k', 100),
            'top_p': gen_params.get('top_p', 1.0),
            'min_p': gen_params.get('min_p', 0.1),
            'rep_pen': gen_params.get('rep_pen', 1.01),
            **result
        }
        enhanced_results.append(enhanced_result)
    
    fieldnames = enhanced_results[0].keys()
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(enhanced_results)
    
    print(f"Results summary saved to: {output_file}")
    return output_file

def save_generation_outputs(results_dir: Path, metadata: dict, model_id: Optional[str] = None):
    """ Save all LLM generation outputs to a separate file """
    # Generate filename for generation outputs
    experiment_meta = metadata.get('experiment_metadata', {})
    model_name = experiment_meta.get('model_name', 'unknown')
    text_name = experiment_meta.get('source_text_name', 'unknown')
    
    filename = generate_results_filename(model_name, text_name, model_id) + "_generations.txt"
    output_file = results_dir / filename
    
    # Load all context results and extract generations
    context_files = list(results_dir.glob("context_*_results.json"))
    if not context_files:
        return
    
    context_results = []
    for filepath in sorted(context_files):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            context_results.append(data)
    
    # Sort by context length
    context_results.sort(key=lambda x: x['context_length'])
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("LLM GENERATION OUTPUTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Source Text: {text_name}\n")
        if model_id:
            f.write(f"Model ID: {model_id}\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("=" * 80 + "\n\n")
        
        for data in context_results:
            context_length = data['context_length']
            rounds = data.get('individual_rounds', [])
            
            f.write(f"CONTEXT LENGTH: {context_length:,} tokens\n")
            f.write("-" * 60 + "\n")
            
            if not rounds:
                f.write("No generation outputs available.\n\n")
                continue
            
            for round_data in rounds:
                round_num = round_data.get('round_number', '?')
                continuation = round_data.get('continuation_text', '')
                timestamp = round_data.get('timestamp', '')
                
                f.write(f"Round {round_num} ({timestamp}):\n")
                f.write("-" * 30 + "\n")
                f.write(continuation)
                f.write("\n" + "-" * 30 + "\n\n")
            
            f.write("\n")
    
    print(f"Generation outputs saved to: {output_file}")
    return output_file

# ================================
# MAIN ORCHESTRATION
# ================================

def main():
    parser = argparse.ArgumentParser(
        description="Test LLM readability degradation across context lengths with incremental state saving",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py novel.txt --api-url http://localhost:5001
  python main.py document.pdf --max-context 16384 --model-name "MyModel"
  python main.py text.txt --word-list dale_chall_words.txt --rounds 3
  python main.py novel.txt --rounds 5 --divisions 2 --temp 0.8 --top-k 50
  python main.py text.txt --max-tokens 1024 --rep-pen 1.05 --min-p 0.05
  python main.py novel.txt --start-context 8192  # Skip testing small contexts
  python main.py --analyze results/model-text-20241201-123456  # Analyze existing results
  python main.py --reanalyze results/model-text-20241201-123456  # Reanalyze stored continuations
"""
    )
    
    parser.add_argument(
        'input_file',
        nargs='?',
        help='Path to reference text file (any format supported by extractous) OR results directory for analysis/reanalysis'
    )
    
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Run analysis on existing results directory'
    )
    
    parser.add_argument(
        '--reanalyze', 
        action='store_true',
        help='Re-run text analysis on stored continuation texts from existing results directory'
    )
    
    parser.add_argument(
        '--api-url',
        default='http://localhost:5001',
        help='API URL for the LLM service'
    )
    
    parser.add_argument(
        '--api-password',
        default=None,
        help='API key/password if required'
    )
    
    parser.add_argument(
        '--word-list',
        default='easy_words.txt',
        help='Path to Dale-Chall easy words list'
    )
    
    parser.add_argument(
        '--max-context',
        type=int,
        default=None,
        help='Maximum context length to test (auto-detect if not specified)'
    )
    
    parser.add_argument(
        '--rounds',
        type=int,
        default=3,
        help='Number of test rounds per context length (default: 3)'
    )
    
    parser.add_argument(
        '--divisions',
        type=int,
        default=1,
        help='Number of context divisions between tiers as a power of 2'
    )
    
    parser.add_argument(
        '--model-name',
        default=None,
        help='Override model name (auto-detected if not provided)'
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=512,
        help='Maximum tokens to generate (default: 1024)'
    )
    
    parser.add_argument(
        '--temp',
        type=float,
        default=1.0,
        help='Generation temperature (default: 1.0)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=100,
        help='Top-k sampling (default: 100)'
    )
    
    parser.add_argument(
        '--top-p',
        type=float,
        default=1.0,
        help='Top-p sampling (default: 1.0)'
    )
    
    parser.add_argument(
        '--min-p',
        type=float,
        default=0.1,
        help='Min-p sampling (default: 0.1)'
    )
    
    parser.add_argument(
        '--rep-pen',
        type=float,
        default=1.01,
        help='Repetition penalty (default: 1.01)'
    )
    
    parser.add_argument(
        '--start-context',
        type=int,
        default=None,
        help='Starting context size in tokens (skip smaller sizes)'
    )
    
    parser.add_argument(
        '--model-id',
        default=None,
        help='Optional model identifier for result filenames (e.g., "v2", "fine-tuned")'
    )
    
    parser.add_argument(
        '--max-retries',
        type=int,
        default=2,
        help='Maximum retries for outlier contexts (default: 2)'
    )
    
    args = parser.parse_args()
    if args.reanalyze:
        if not args.input_file or not Path(args.input_file).exists():
            print("Error: Must specify valid results directory for reanalysis")
            return 1
        
        try:
            results_dir = Path(args.input_file)
            client = StreamingAPIClient(args.api_url, args.api_password)
            initialize_word_list(args.word_list)
            reanalyze(results_dir, client, args.model_id)
            return 0
        except Exception as e:
            print(f"Reanalysis failed: {e}")
            return 1
        
    if args.analyze:
        if not args.input_file or not Path(args.input_file).exists():
            print("Error: Must specify valid results directory for analysis")
            return 1
        
        try:
            results_dir = Path(args.input_file)
            run_analysis(results_dir, args.model_id)
            return 0
        except Exception as e:
            print(f"Analysis failed: {e}")
            return 1
    
    if not args.input_file:
        print("Error: Must specify input file for data collection")
        return 1
    
    if not is_power_of_two(args.divisions):
        print(f"Divisions must be 1 or a power of 2 such as 2 or 4 or 8")
        return 1
    
    try:
        # Initialize components
        print("=" * 60)
        print("MODEL READABILITY DEGRADATION TEST - INCREMENTAL STATE SAVING")
        print("=" * 60)
        
        client = StreamingAPIClient(args.api_url, args.api_password)
        initialize_word_list(args.word_list)
        
        model_name = args.model_name or client.get_model_name() or "unknown-model"
        text_name = os.path.basename(args.input_file)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Create output directory
        output_dir = create_output_directory(model_name, text_name, timestamp)
        print(f"Output directory: {output_dir}")
        
        # Prepare experiment metadata
        metadata = {
            'experiment_metadata': {
                'model_name': model_name,
                'model_id': args.model_id,
                'source_text_file': args.input_file,
                'source_text_name': text_name,
                'start_time': datetime.now().isoformat(),
                'num_rounds': args.rounds,
                'divisions': args.divisions,
                'start_context': args.start_context,
                'max_retries': args.max_retries,
                'generation_params': {
                    'max_tokens': args.max_tokens,
                    'temperature': args.temp,
                    'top_k': args.top_k,
                    'top_p': args.top_p,
                    'min_p': args.min_p,
                    'rep_pen': args.rep_pen
                },
                'api_url': client.api_url
            }
        }
        
        # Load and prepare text
        reference_text = load_reference_text(args.input_file)
        if args.max_context is not None:
            max_context = min(args.max_context, client.get_max_context_length())
        else:
            max_context = client.get_max_context_length()
        metadata['experiment_metadata']['max_context'] = max_context
        
        context_lengths = generate_context_lengths(max_context, args.divisions, args.start_context)
        metadata['context_lengths'] = context_lengths
        
        working_tokens = prepare_working_tokens(reference_text, max_context, client)
        # Analyze baseline
        #print(f"\nGenerating embeddings for sample text...")
        #baseline_text = client.tokens_to_text(working_tokens[-args.max_tokens:] if len(working_tokens) > args.max_tokens else working_tokens)
        
        #baseline_analysis = analyze_text_comprehensive(baseline_text, client)
        #print(f"\nBaseline text Analysis (from continuation point area):")
        
        #print(f"  Cloze Score: {baseline_analysis['cloze_score']}")
        #print(f"  Reading Level: {baseline_analysis['reading_level']}")
        #print(f"  Vocabulary Diversity: {baseline_analysis['vocabulary_diversity']}")
        
        #metadata['baseline_analysis'] = baseline_analysis
        
        # Save initial metadata
        save_experiment_metadata(output_dir, metadata)
        
        # Run benchmarks
        print(f"\n{'='*60}")
        print("RUNNING DEGRADATION TESTS")
        print(f"All continuations start from same story position!")
        print(f"{'='*60}")
        
        generation_params = {
            'max_tokens': args.max_tokens,
            'temperature': args.temp,
            'top_k': args.top_k,
            'top_p': args.top_p,
            'min_p': args.min_p,
            'rep_pen': args.rep_pen
        }
        
        successful_contexts = 0
        
        for i, context_length in enumerate(context_lengths, 1):
            print(f"\n[TEST {i}/{len(context_lengths)}] Context Length: {context_length:,} tokens")
            print("-" * 50)
            
            # Build context window
            try:
                context = build_context_window(working_tokens, context_length, client, args.max_tokens)
            except Exception as e:
                print(f"WARNING: Failed to build context for {context_length} tokens: {e}")
                continue
            
            if not context:
                print(f"WARNING: Failed to build context for {context_length} tokens")
                continue
            
            context_info = {
                'context_length_target': context_length,
                'context_length_actual': client.count_tokens(context),
                'context_text_chars': len(context),
                'context_word_count': len(context.split()),
                'context_preview': context[-500:] if len(context) > 500 else context
            }
            
            print(f"Context built successfully ({len(context.split())} words)")
            
            # Run benchmarks with retry logic for outliers
            retry_count = 0
            while retry_count <= args.max_retries:
                if retry_count > 0:
                    print(f"\nRetry {retry_count}/{args.max_retries} for context {context_length:,}")
                
                results = run_benchmarks(context, args.rounds, client, generation_params)
                
                if not results:
                    print(f"WARNING: No successful rounds for context length {context_length}")
                    break
                
                # Check for outliers
                outlier_indices = check_for_outliers(results)
                
                if not outlier_indices:
                    # No outliers, save results and continue
                    save_context_results(output_dir, context_length, context_info, results, generation_params)
                    
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
                    
                    successful_contexts += 1
                    break
                else:
                    print(f"Outliers detected at indices: {outlier_indices}")
                    retry_count += 1
                    
                    if retry_count > args.max_retries:
                        print(f"Max retries exceeded. Skipping context {context_length:,} due to persistent outliers.")
                        break
        
        # Update metadata with completion info
        metadata['experiment_metadata']['end_time'] = datetime.now().isoformat()
        metadata['experiment_metadata']['successful_contexts'] = successful_contexts
        save_experiment_metadata(output_dir, metadata)
        
        print(f"\n{'='*60}")
        print("DATA COLLECTION COMPLETE")
        print(f"{'='*60}")
        print(f"Successfully tested {successful_contexts}/{len(context_lengths)} context lengths")
        print(f"Results saved to: {output_dir}")
        print(f"\nTo analyze results, run:")
        print(f"  python {__file__} --analyze {output_dir}")
        
        # Auto-run analysis if we have sufficient data
        if successful_contexts >= 2:
            print(f"\nRunning automatic analysis...")
            run_analysis(output_dir, args.model_id)
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
