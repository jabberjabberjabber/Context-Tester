#!/usr/bin/env python3
"""
File I/O operations for Context Tester.

Handles all file reading, writing, and result management.
"""

import json
import csv
import statistics
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


def create_output_directory(model_name: str, text_name: str, model_id: str) -> Path:
    """Create output directory for experiment results.

    Args:
        model_name: Full model name (e.g., "deepseek-ai/deepseek-r1-0528")
        text_name: Source text filename (e.g., "middlemarch.txt")
        model_id: Model identifier (timestamp or custom ID)

    Returns:
        Path to created output directory

    Raises:
        FileExistsError: If directory already exists (prevents overwriting)
    """
    # Keep organization in directory name
    clean_model = _normalize_filename_part(model_name)
    # Remove extension from text name
    text_base = Path(text_name).stem
    clean_text = _normalize_filename_part(text_base)
    clean_id = _normalize_filename_part(model_id)

    # Format: org-model-text-id (e.g., "deepseek-ai-deepseek-r1-0528-middlemarch-20250108-123456")
    dir_name = f"{clean_model}-{clean_text}-{clean_id}"
    output_dir = Path("results") / dir_name

    # Check if directory already exists
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_dir}\n"
            f"This likely means you're reusing a model_id.\n"
            f"Either:\n"
            f"  1. Use a different --model-id\n"
            f"  2. Delete the existing directory\n"
            f"  3. Don't specify --model-id to auto-generate timestamp"
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def save_experiment_metadata(output_dir: Path, metadata: dict):
    """Save experiment metadata to JSON file."""
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def load_experiment_metadata(results_dir: Path) -> dict:
    """Load experiment metadata from results directory."""
    metadata_file = results_dir / "metadata.json"
    if not metadata_file.exists():
        raise ValueError(f"No metadata found in {results_dir}")

    with open(metadata_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_context_results(
    output_dir: Path,
    context_length: int,
    context_info: dict,
    results: List[dict],
    generation_params: dict
):
    """Save results for a single context length."""
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


def load_all_context_results(results_dir: Path) -> List[dict]:
    """Load all context result files from directory."""
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
    return context_results


def load_individual_rounds_for_plotting(results_dir: Path) -> tuple[List[dict], List[dict], dict]:
    """Load individual rounds and averaged results from results directory.

    Args:
        results_dir: Path to results directory

    Returns:
        Tuple of (individual_rounds_data, averaged_results, metadata)
        - individual_rounds_data: List of dicts with individual round data by context length
        - averaged_results: List of averaged results by context length
        - metadata: Experiment metadata
    """
    context_results = load_all_context_results(results_dir)
    metadata = load_experiment_metadata(results_dir)

    individual_rounds_data = []
    averaged_results = []

    for context_data in context_results:
        context_length = context_data['context_length']
        rounds = context_data.get('individual_rounds', [])
        averaged = context_data.get('averaged_stats', {})

        # Add context_length to averaged stats
        if averaged:
            averaged['context_length'] = context_length
            averaged_results.append(averaged)

        # Store individual rounds with context_length
        for round_data in rounds:
            round_with_context = round_data.copy()
            round_with_context['context_length'] = context_length
            individual_rounds_data.append(round_with_context)

    return individual_rounds_data, averaged_results, metadata


def average_results(round_results: List[Dict]) -> Dict[str, Any]:
    """Average results across multiple rounds."""
    from readability_tests import reading_level_from_cloze

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

    # Copy non-numerical fields from first result
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

    # Calculate standard deviations
    cloze_values = [r['cloze_score'] for r in round_results if 'cloze_score' in r and r['cloze_score'] is not None]
    vocab_values = [r['vocabulary_diversity'] for r in round_results if 'vocabulary_diversity' in r and r['vocabulary_diversity'] is not None]

    averaged['cloze_score_std'] = round(statistics.stdev(cloze_values), 3) if len(cloze_values) > 1 else 0.0
    averaged['vocab_diversity_std'] = round(statistics.stdev(vocab_values), 4) if len(vocab_values) > 1 else 0.0

    return averaged


def save_results_csv(results: List[dict], output_dir: Path, metadata: dict, model_id: Optional[str] = None) -> Path:
    """Save results to CSV for plotting."""
    if not results:
        return None

    # Generate filename
    experiment_meta = metadata.get('experiment_metadata', {})
    model_name = experiment_meta.get('model_name', 'unknown')
    text_name = experiment_meta.get('source_text_name', 'unknown')

    # Use model_id from metadata if not explicitly passed
    if model_id is None:
        model_id = experiment_meta.get('model_id', '')

    filename = _generate_results_filename(model_name, text_name, model_id) + ".csv"
    output_file = output_dir / filename

    # Add metadata to each result
    enhanced_results = []
    gen_params = experiment_meta.get('generation_params', {})

    for result in results:
        enhanced_result = {
            'model_name': model_name,
            'text_name': text_name,
            'model_id': model_id,
            'max_tokens': gen_params.get('max_tokens', 1024),
            'temperature': gen_params.get('temperature', 1.0),
            'top_k': gen_params.get('top_k', 100),
            'top_p': gen_params.get('top_p', 1.0),
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


def save_generation_outputs(results_dir: Path, metadata: dict, model_id: Optional[str] = None) -> Path:
    """Save all LLM generation outputs to a text file."""
    experiment_meta = metadata.get('experiment_metadata', {})
    model_name = experiment_meta.get('model_name', 'unknown')
    text_name = experiment_meta.get('source_text_name', 'unknown')

    filename = _generate_results_filename(model_name, text_name, model_id) + "_generations.txt"
    output_file = results_dir / filename

    # Load all context results
    context_results = load_all_context_results(results_dir)

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


def save_analysis_results(results_dir: Path, analysis: dict):
    """Save degradation analysis results."""
    analysis_file = results_dir / "degradation_analysis.json"
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)


def _normalize_filename_part(text: str) -> str:
    """Normalize text for safe filename usage."""
    import re
    normalized = text.replace('/', '-').replace('\\', '-').replace(':', '-')
    normalized = re.sub(r'[<>"|?*]', '', normalized)
    normalized = re.sub(r'\s+', '_', normalized.strip())
    return normalized


def _generate_results_filename(model_name: str, text_name: str, model_id: Optional[str] = None) -> str:
    """Generate results filename from model name, text name, and optional model ID."""
    clean_model = _normalize_filename_part(model_name)
    clean_text = _normalize_filename_part(text_name)

    if model_id and model_id.strip():
        clean_id = _normalize_filename_part(model_id.strip())
        return f"{clean_model}-{clean_text}-{clean_id}"
    else:
        return f"{clean_model}-{clean_text}"


# ================================
# DATASET NAME EXTRACTION
# ================================

def get_dataset_name_from_csv(csv_file: Path) -> str:
    """Extract dataset name from CSV metadata columns.

    Tries to extract model_name, text_name, and model_id from the CSV.
    Falls back to filename if metadata not available.

    Args:
        csv_file: Path to CSV file

    Returns:
        Clean dataset name in format: "model_name text_name model_id"
    """
    try:
        df = pd.read_csv(csv_file, nrows=1)  # Read just first row for metadata

        model_name = df['model_name'].iloc[0] if 'model_name' in df.columns else None
        text_name = df['text_name'].iloc[0] if 'text_name' in df.columns else None
        model_id = df['model_id'].iloc[0] if 'model_id' in df.columns else None

        # Strip organization prefix from model name (e.g., "deepseek-ai/deepseek-r1" -> "deepseek-r1")
        if model_name and str(model_name) != 'nan':
            model_name_str = str(model_name)
            if '/' in model_name_str:
                model_name_str = model_name_str.split('/')[-1]
            model_name = model_name_str

        # Build name from available components
        parts = []
        if model_name and str(model_name) != 'nan':
            parts.append(str(model_name))
        if text_name and str(text_name) != 'nan':
            parts.append(str(text_name))
        if model_id and str(model_id) != 'nan' and str(model_id) != '':
            parts.append(str(model_id))

        if parts:
            return ' '.join(parts)

    except Exception:
        pass

    # Fallback to filename
    return Path(csv_file).stem


def load_csv_for_plotting(csv_file: Path) -> tuple[pd.DataFrame, str]:
    """Load CSV file and extract dataset name.

    Args:
        csv_file: Path to CSV file

    Returns:
        Tuple of (dataframe, dataset_name)
    """
    df = pd.read_csv(csv_file)
    # Filter out null context lengths
    df_clean = df[df['context_length'].notna()].copy()
    dataset_name = get_dataset_name_from_csv(csv_file)
    return df_clean, dataset_name


def generate_plot_filename(dataset_names: List[str]) -> str:
    """Generate output PNG filename based on dataset names.

    Args:
        dataset_names: List of dataset names from CSV metadata

    Returns:
        Generated PNG filename
    """
    # Clean dataset names for filenames (replace spaces with underscores, remove special chars)
    def clean_for_filename(name):
        return name.replace(' ', '_').replace('/', '-').replace('\\', '-').replace(':', '-')

    cleaned_names = [clean_for_filename(name) for name in dataset_names]

    if len(cleaned_names) == 1:
        return f"{cleaned_names[0]}.png"

    # Multiple files: model1_text1_id1_with_model2_text2_id2.png
    return f"{cleaned_names[0]}_with_{'_'.join(cleaned_names[1:])}.png"
