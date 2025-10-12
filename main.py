#!/usr/bin/env python3
"""
Model Readability Degradation Test - Streamlined Architecture

Tests LLM output quality degradation at increasing context lengths by analyzing
the readability complexity of generated text continuations using Cloze scores.

Refactored for maintainability with modular architecture.
"""

import os
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from extractous import Extractor

# Local modules
from config import parse_arguments, create_generation_params, create_experiment_metadata
from src.file_operations import (
    create_output_directory,
    save_experiment_metadata,
    save_context_results,
    load_experiment_metadata,
    load_all_context_results,
    save_results_csv,
    save_generation_outputs,
    save_analysis_results
)
from src.streaming_api import StreamingAPIClient
from benchmark_runner import run_context_test_with_retries
from src.readability_tests import initialize_word_list
from generate_plot import make_png


# ================================
# TEXT PROCESSING
# ================================

def normalize_content(content: str) -> str:
    """Convert fixed-width text and normalize characters."""
    content = unicodedata.normalize('NFKC', content)
    content = content.replace('--', '—')
    content = content.replace('"', '"').replace('"', '"')
    content = content.replace(''', "'").replace(''', "'")

    text = content.replace('\r\n', '\n').replace('\r', '\n')
    paragraphs = text.split('\n\n')

    result = '\n\n'.join(para.replace('\n', ' ') for para in paragraphs)
    #result = result.replace('\n\n', '\n\n    ')
    return result


def load_reference_text(file_path: str) -> str:
    """Load reference text from file (supports txt, pdf, html)."""
    extractor = Extractor()

    content, metadata = extractor.extract_file_to_string(file_path)
    print(f"Loaded reference text: {metadata.get('resourceName', file_path)}")
    print(f"Content type: {metadata.get('Content-Type', 'Unknown')}")

    if not content or not content.strip():
        raise ValueError(f"File {file_path} appears to be empty or could not be read")

    content_size = int(len(content) * 0.8)
    normalized_content = normalize_content(content[:content_size])
    print(f"Text preview (first 200 chars): {normalized_content[:200]!r}")
    print(f"Total characters: {len(normalized_content):,}")
    return normalized_content


def prepare_working_tokens(text: str, max_context: int, client: StreamingAPIClient) -> List[str]:
    """
    Tokenize text and prepare working token set.

    Args:
        text: Reference text to tokenize
        max_context: Maximum context length we'll test
        client: API client for tokenization

    Returns:
        List of tokens for testing
    """
    min_required = int(max_context * 1.2)

    all_tokens = client.tokenize_text_batched(text)
    if not all_tokens:
        print(f"No tokenizer available. Exiting.")
        sys.exit(1)

    if len(all_tokens) < min_required:
        # Handle repetition case
        repeats_needed = (min_required // len(all_tokens)) + 1
        working_tokens = (all_tokens * repeats_needed)[:min_required]
    else:
        working_tokens = all_tokens[:min_required]

    # Convert back to text and prune at natural boundaries
    working_text = client.tokens_to_text_batched(working_tokens)
    pruned_text = client.prune_text(working_text, max_context, len(working_tokens))

    # Final tokenization
    final_tokens = client.tokenize_text_batched(pruned_text)
    return final_tokens


# ================================
# CONTEXT GENERATION
# ================================

def generate_context_lengths(
    max_context: int,
    divisions: int = 1,
    start_context: Optional[int] = None
) -> List[int]:
    """
    Generate power-of-2 context lengths to test with subdivisions.

    Args:
        max_context: Maximum context length
        divisions: Number of splits per power of 2 tier
        start_context: Optional starting context size (skip smaller sizes)

    Returns:
        List of context lengths in tokens
    """
    # Generate base powers of 2: 2k, 4k, 8k, 16k, 32k, etc.
    tiers = []
    power = 11  # Start at 2^11 = 2048

    while True:
        length = 2 ** power
        if length > max_context:
            break

        tiers.append(length)
        power += 1

    if not tiers:
        return [min(1024, max_context)]

    # Filter tiers based on start_context
    if start_context is not None:
        tiers = [tier for tier in tiers if tier >= start_context]
        if not tiers:
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


def build_context_window(
    working_tokens: List[str],
    context_length: int,
    client: StreamingAPIClient,
    max_tokens: int
) -> str:
    """Build context window from working tokens using backward expansion."""
    # Calculate overhead tokens
    system_message = "You are a skilled novelist continuing a story."
    instruction = """Continue this story for as long as you can. Do not try to add a conclusion or ending, just keep writing as if this were part of the middle of a novel. Maintain the same style, tone, and narrative voice. Focus on developing the plot, characters, and setting naturally."""

    overhead_tokens = client.count_tokens(system_message) + client.count_tokens(instruction) + 20

    adjusted_length = context_length - max_tokens - overhead_tokens

    print(f"  Reserved {overhead_tokens} tokens for overhead, {max_tokens} for completion")

    # Convert tokens to text
    target_tokens = working_tokens[-adjusted_length:]
    context_text = client.tokens_to_text_batched(target_tokens)

    if not context_text:
        return None

    print(f"  Context window: {adjusted_length:,} tokens")
    return context_text


# ================================
# ANALYSIS
# ================================

def analyze_degradation_trends(results: List[dict], metadata: dict) -> dict:
    """Analyze degradation trends across context lengths."""
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


def print_results_table(results: List[dict]):
    """Print formatted results table."""
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


def run_analysis(results_dir: Path, model_id: Optional[str] = None) -> dict:
    """
    Load all saved results and perform degradation analysis.

    Args:
        results_dir: Directory containing saved context results
        model_id: Optional model identifier for filename generation

    Returns:
        Analysis results dictionary
    """
    print(f"\n{'='*60}")
    print("RUNNING DEGRADATION ANALYSIS")
    print(f"{'='*60}")

    metadata = load_experiment_metadata(results_dir)
    context_results = load_all_context_results(results_dir)

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
    save_analysis_results(results_dir, analysis)

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


def reanalyze(input_dir: Path, client: StreamingAPIClient, model_id: Optional[str] = None) -> Path:
    """
    Re-analyze existing results using stored continuation texts.

    Args:
        input_dir: Directory containing context_*_results.json files
        client: API client for token counting and text analysis
        model_id: Optional model identifier for new result filenames

    Returns:
        Path to new output directory with reanalyzed results
    """
    from src.readability_tests import analyze_text_comprehensive

    print(f"\n{'='*60}")
    print("RE-ANALYZING EXISTING RESULTS")
    print(f"{'='*60}")
    print(f"Input directory: {input_dir}")

    metadata = load_experiment_metadata(input_dir)
    context_files = list(input_dir.glob("context_*_results.json"))

    if not context_files:
        raise ValueError(f"No context results found in {input_dir}")

    # Create new output directory
    experiment_meta = metadata.get('experiment_metadata', {})
    model_name = experiment_meta.get('model_name', 'unknown')
    text_name = experiment_meta.get('source_text_name', 'unknown')

    # Generate new model_id for reanalysis
    reanalysis_id = model_id if model_id else f"reanalyzed-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    output_dir = create_output_directory(model_name, text_name, reanalysis_id)
    print(f"Output directory: {output_dir}")

    # Update metadata
    metadata['reanalysis_metadata'] = {
        'original_directory': str(input_dir),
        'reanalysis_time': datetime.now().isoformat(),
        'reanalysis_model_id': reanalysis_id,
        'analysis_method': 'reanalyzed_from_stored_continuations'
    }
    metadata['experiment_metadata']['model_id'] = reanalysis_id

    print(f"Found {len(context_files)} context result files to reanalyze")

    successful_contexts = 0

    for context_file in sorted(context_files):
        print(f"\nReanalyzing: {context_file.name}")

        with open(context_file, 'r', encoding='utf-8') as f:
            import json
            context_data = json.load(f)

        context_length = context_data['context_length']
        context_info = context_data['context_info']
        generation_params = context_data['generation_params']
        original_rounds = context_data.get('individual_rounds', [])

        if not original_rounds:
            print(f"  No rounds found, skipping")
            continue

        print(f"  Context length: {context_length:,} tokens")
        print(f"  Reanalyzing {len(original_rounds)} rounds...")

        reanalyzed_rounds = []

        for round_data in original_rounds:
            continuation_text = round_data.get('continuation_text', '')

            if not continuation_text:
                continue

            try:
                new_analysis = analyze_text_comprehensive(continuation_text, client)

                new_round_data = {
                    'round_number': round_data.get('round_number'),
                    'continuation_length': len(continuation_text),
                    'continuation_tokens': round_data.get('continuation_tokens'),
                    'generation_time_seconds': round_data.get('generation_time_seconds'),
                    'timestamp': round_data.get('timestamp'),
                    'continuation_text': continuation_text,
                    'reanalyzed_timestamp': datetime.now().isoformat(),
                    **new_analysis
                }

                reanalyzed_rounds.append(new_round_data)

            except Exception as e:
                print(f"    Failed to reanalyze round {round_data.get('round_number', '?')}: {e}")
                continue

        if reanalyzed_rounds:
            save_context_results(output_dir, context_length, context_info, reanalyzed_rounds, generation_params)
            successful_contexts += 1

    metadata['reanalysis_metadata']['successful_contexts'] = successful_contexts
    metadata['reanalysis_metadata']['total_contexts'] = len(context_files)

    save_experiment_metadata(output_dir, metadata)

    print(f"\n{'='*60}")
    print("REANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Successfully reanalyzed {successful_contexts}/{len(context_files)} context lengths")

    if successful_contexts >= 2:
        run_analysis(output_dir, model_id)

    return output_dir


# ================================
# MAIN ORCHESTRATION
# ================================

def run_data_collection(args):
    """Run data collection mode."""
    print("=" * 60)
    print("MODEL READABILITY DEGRADATION TEST")
    print("=" * 60)

    # Get HF token from environment
    hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')

    # Initialize client
    client = StreamingAPIClient(
        args.api_url,
        args.api_password,
        tokenizer_model=args.tokenizer_model,
        model_name=args.model_name,
        max_context=args.max_context,
        embedding_model=args.embedding_model,
        hf_token=hf_token
    )

    initialize_word_list(args.word_list)

    model_name = args.model_name or client.get_model_name() or "unknown-model"
    text_name = os.path.basename(args.input_file)

    # Ensure model_id is always set (use timestamp if not provided)
    model_id = args.model_id if args.model_id else datetime.now().strftime("%Y%m%d-%H%M%S")

    # Create metadata with guaranteed model_id
    metadata = create_experiment_metadata(args, model_name, text_name)
    metadata['experiment_metadata']['model_id'] = model_id

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

    # Prepare output directory path (but don't create it yet)
    output_dir = None
    output_dir_path = None

    # Run benchmarks
    print(f"\n{'='*60}")
    print("RUNNING DEGRADATION TESTS")
    print(f"All continuations start from same story position!")
    print(f"{'='*60}")

    generation_params = create_generation_params(args)
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
            break

        context_info = {
            'context_length_target': context_length,
            'context_length_actual': client.count_tokens(context),
            'context_text_chars': len(context),
            'context_word_count': len(context.split()),
            'context_preview': context[-500:] if len(context) > 500 else context
        }

        print(f"Context built successfully ({len(context.split())} words)")

        # Create output directory on first successful context (lazy initialization)
        if output_dir is None:
            output_dir = create_output_directory(model_name, text_name, model_id)
            print(f"\nCreating output directory: {output_dir}")
            # Save initial metadata now that we know we'll have data
            save_experiment_metadata(output_dir, metadata)

        # Run test with retries
        success = run_context_test_with_retries(
            context,
            context_length,
            context_info,
            args.rounds,
            client,
            generation_params,
            args.max_retries,
            args.ignore_min_tokens,
            output_dir
        )

        if success:
            successful_contexts += 1

    # Update metadata with completion info (only if we created output dir)
    if output_dir is not None:
        metadata['experiment_metadata']['end_time'] = datetime.now().isoformat()
        metadata['experiment_metadata']['successful_contexts'] = successful_contexts
        save_experiment_metadata(output_dir, metadata)

    print(f"\n{'='*60}")
    print("DATA COLLECTION COMPLETE")
    print(f"{'='*60}")

    if successful_contexts == 0:
        print("WARNING: No successful test runs completed.")
        print("No output directory was created.")
        print("\nPossible issues:")
        print("  - API connection problems")
        print("  - Tokenizer initialization failed")
        print("  - Context window building errors")
        return

    print(f"Successfully tested {successful_contexts}/{len(context_lengths)} context lengths")
    print(f"Results saved to: {output_dir}")
    print(f"\nTo analyze results, run:")
    print(f"  python {__file__} --analyze {output_dir}")

    # Auto-run analysis if we have sufficient data
    if successful_contexts >= 2:
        print(f"\nRunning automatic analysis...")
        run_analysis(output_dir, args.model_id)


def main():
    """Main entry point."""
    try:
        args = parse_arguments()

        if args.reanalyze:
            results_dir = Path(args.input_file)
            hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
            client = StreamingAPIClient(
                args.api_url,
                args.api_password,
                tokenizer_model=args.tokenizer_model,
                model_name=args.model_name,
                max_context=args.max_context,
                embedding_model=args.embedding_model,
                hf_token=hf_token
            )
            initialize_word_list(args.word_list)
            reanalyze(results_dir, client, args.model_id)
            return 0

        if args.analyze:
            results_dir = Path(args.input_file)
            run_analysis(results_dir, args.model_id)
            return 0

        # Data collection mode
        run_data_collection(args)
        return 0

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
