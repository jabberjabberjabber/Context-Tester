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
    """Average results across multiple rounds.

    Automatically detects numerical fields and averages them.
    Excludes specific fields that should not be averaged.
    """
    from src.readability_tests import reading_level_from_cloze

    if not round_results:
        return {}

    if len(round_results) == 1:
        result = round_results[0].copy()
        result['num_rounds'] = 1
        result['cloze_score_std'] = 0.0
        result['vocab_diversity_std'] = 0.0
        return result

    # Fields that should NOT be averaged (either non-numerical or identifiers)
    exclude_fields = {
        'round_number',           # Round identifier
        'timestamp',              # Timestamp string
        'continuation_text',      # Generated text
        'reading_level',          # String derived from cloze_score
        'reanalyzed_timestamp',   # Timestamp string
        'context_length'          # Context identifier (same across all rounds at this level)
    }

    averaged = {}

    # Auto-detect numerical fields from first result
    numerical_fields = []
    for key, value in round_results[0].items():
        if key not in exclude_fields and isinstance(value, (int, float)) and value is not None:
            numerical_fields.append(key)

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

        # Include ground truth if available
        ground_truth = metadata.get('ground_truth_analysis', {})
        if ground_truth:
            f.write("\n" + "=" * 80 + "\n")
            f.write("GROUND TRUTH (Source Text Continuation)\n")
            f.write("=" * 80 + "\n")
            f.write("This is the actual continuation from the source text at the continuation point.\n")
            f.write("Use this as a baseline for comparison with generated outputs.\n\n")

            # Get ground truth text from first context (they should all have the same ground truth)
            ground_truth_text = None
            for data in context_results:
                context_info = data.get('context_info', {})
                if 'ground_truth_text' in context_info:
                    ground_truth_text = context_info['ground_truth_text']
                    break

            if ground_truth_text:
                f.write(f"Ground Truth Metrics:\n")
                f.write(f"  Cloze Score: {ground_truth.get('cloze_score', 'N/A')}\n")
                f.write(f"  Vocabulary Diversity: {ground_truth.get('vocabulary_diversity', 'N/A')}\n")
                f.write(f"  Reading Level: {ground_truth.get('reading_level', 'N/A')}\n")
                f.write(f"  Avg Sentence Length: {ground_truth.get('avg_sentence_length', 'N/A')}\n")
                f.write("-" * 60 + "\n")
                f.write(ground_truth_text)
                f.write("\n" + "=" * 80 + "\n\n")

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


def load_all_results_from_folder(parent_folder: Path) -> List[Dict[str, Any]]:
    """Load all valid result directories from a parent folder.

    Args:
        parent_folder: Path to folder containing result directories

    Returns:
        List of dicts, each containing:
            - 'name': Display name for the dataset
            - 'dataframe': pandas DataFrame with averaged results
            - 'metadata': Full metadata dict
            - 'error': Error message if loading failed (optional)
    """
    datasets = []

    for subdir in parent_folder.iterdir():
        if not subdir.is_dir():
            continue

        # Check if it's a valid results directory
        if not (subdir / "metadata.json").exists():
            continue

        try:
            # Load metadata
            metadata = load_experiment_metadata(subdir)
            experiment_meta = metadata.get('experiment_metadata', {})

            model_name = experiment_meta.get('model_name', 'unknown')
            text_name = experiment_meta.get('source_text_name', 'unknown')
            model_id = experiment_meta.get('model_id', '')

            # Clean up model name
            if '/' in str(model_name):
                model_name = str(model_name).split('/')[-1]

            # Build display name
            parts = [str(model_name), str(text_name)]
            if model_id and str(model_id) != '':
                parts.append(str(model_id))
            name = ' - '.join(parts)

            # Load data
            individual_rounds, averaged_results, _ = load_individual_rounds_for_plotting(subdir)

            # Create dataframe
            df = pd.DataFrame(averaged_results)
            df = df[df['context_length'].notna()].copy()

            if df.empty:
                datasets.append({
                    'name': name,
                    'error': 'No valid data in results'
                })
                continue

            datasets.append({
                'name': name,
                'dataframe': df,
                'metadata': metadata
            })

        except Exception as e:
            datasets.append({
                'name': subdir.name,
                'error': str(e)
            })

    return datasets


def generate_plot_filename(dataset_names: List[str]) -> str:
    """Generate output PNG filename based on dataset names.

    Args:
        dataset_names: List of dataset names from CSV metadata

    Returns:
        Generated PNG filename (max 200 chars to avoid Windows path limits)
    """
    # Clean dataset names for filenames (replace spaces with underscores, remove special chars)
    def clean_for_filename(name):
        return name.replace(' ', '_').replace('/', '-').replace('\\', '-').replace(':', '-')

    cleaned_names = [clean_for_filename(name) for name in dataset_names]

    if len(cleaned_names) == 1:
        return f"{cleaned_names[0]}.png"

    # Check if this looks like --plot-rounds output (multiple datasets with R1, R2, etc. and AVG)
    # Pattern: "base_name R1", "base_name R2", ..., "base_name AVG"
    if len(cleaned_names) > 3:
        # Check if all names share a common base
        first_parts = [' '.join(name.split()[:-1]) for name in dataset_names]
        last_parts = [name.split()[-1] for name in dataset_names]

        # If they all have the same base and end with R1, R2, etc, AVG
        if len(set(first_parts)) == 1 and 'AVG' in last_parts:
            # This is --plot-rounds output, use base name + "_rounds.png"
            base = clean_for_filename(first_parts[0])
            return f"{base}_rounds.png"

    # Standard comparison: limit to avoid Windows 260 char path limit
    # Use first dataset + count of others
    if len(cleaned_names) > 6:
        return f"{cleaned_names[0]}_with_{len(cleaned_names)-1}_others.png"

    # Multiple files: model1_text1_id1_with_model2_text2_id2.png
    filename = f"{cleaned_names[0]}_with_{'_'.join(cleaned_names[1:])}.png"

    # If still too long, truncate
    if len(filename) > 200:
        return f"{cleaned_names[0]}_with_{len(cleaned_names)-1}_datasets.png"

    return filename
