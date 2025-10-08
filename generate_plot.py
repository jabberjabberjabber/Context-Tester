#!/usr/bin/env python3
"""
Multi-Dataset Performance Comparison Tool

Analyzes and visualizes performance differences between multiple configurations
across different context lengths. Supports 1-6 CSV input files.
Saves plots as PNG files.

Usage:
    python analysis.py dataset1.csv dataset2.csv dataset3.csv
    python analysis.py rope.csv full_context.csv --output comparison.png
    python analysis.py config1.csv config2.csv config3.csv config4.csv --no-plots

    # From another script:
    from generate_plot import make_png
    success = make_png(enhanced_results, output_file_path)

Requirements:
    pip install pandas matplotlib numpy
"""

#################################
#VIBE CODED with Claude Sonnet 4#
#################################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import re
from pathlib import Path
from file_operations import (
    get_dataset_name_from_csv,
    load_csv_for_plotting,
    generate_plot_filename,
    load_individual_rounds_for_plotting,
    load_experiment_metadata
)


def get_dataset_name(filepath, all_paths=None):
    """Extract dataset name from file path (filename without extension).

    DEPRECATED: Use get_dataset_name_from_csv() instead for better labels.

    If multiple files have the same name, includes parent directory to differentiate.

    Args:
        filepath: Path to CSV file
        all_paths: Optional list of all file paths to check for duplicates

    Returns:
        Clean dataset name for use in plots and analysis
    """
    path = Path(filepath)
    base_name = path.stem

    # If we have multiple files, check for duplicate names
    if all_paths and len(all_paths) > 1:
        # Get all base names
        base_names = [Path(p).stem for p in all_paths]

        # If this name appears multiple times, include parent directory
        if base_names.count(base_name) > 1:
            # Use parent directory name as differentiator
            parent_name = path.parent.name
            return f"{parent_name}_{base_name}"

    return base_name

def get_base_name(filename):
    """ Extract base name before first non-alphanumeric character.
    
    Args:
        filename: Filename string
    
    Returns:
        Base name truncated at first non-alphanumeric character
    """
    # Remove extension first
    #base = os.path.splitext(filename)[1]
    #base = Path(filename).stem
    # Find first non-alphanumeric character (excluding underscore and hyphen)
    #match = re.search(r'[^a-zA-Z0-9_-]', base)
    #if match:
    #    return base[:match.start()]
    #print(base)
    return os.path.splitext(os.path.basename(filename))[0]

def generate_output_filename(csv_files, dataset_names=None):
    """ Generate output PNG filename based on dataset names from CSV metadata.

    Args:
        csv_files: List of CSV file paths
        dataset_names: Optional list of dataset names (will be extracted if not provided)

    Returns:
        Generated PNG filename
    """
    if dataset_names is None:
        dataset_names = [get_dataset_name_from_csv(f) for f in csv_files]

    # Use file_operations function
    return generate_plot_filename(dataset_names)

def is_results_directory(path: Path) -> bool:
    """Check if path is a results directory (contains metadata.json)."""
    return path.is_dir() and (path / "metadata.json").exists()


def load_data_from_directory(results_dir: Path, plot_rounds: bool = False):
    """Load data from results directory.

    Args:
        results_dir: Path to results directory
        plot_rounds: If True, return individual round data

    Returns:
        Tuple of (dataframe, dataset_name, rounds_data_if_requested)
    """
    metadata = load_experiment_metadata(results_dir)
    experiment_meta = metadata.get('experiment_metadata', {})

    model_name = experiment_meta.get('model_name', 'unknown')
    text_name = experiment_meta.get('source_text_name', 'unknown')
    model_id = experiment_meta.get('model_id', '')

    # Build dataset name
    parts = []
    if model_name and str(model_name) != 'nan':
        # Strip organization
        if '/' in str(model_name):
            model_name = str(model_name).split('/')[-1]
        parts.append(str(model_name))
    if text_name and str(text_name) != 'nan':
        parts.append(str(text_name))
    if model_id and str(model_id) != 'nan' and str(model_id) != '':
        parts.append(str(model_id))

    dataset_name = ' '.join(parts) if parts else results_dir.name

    if plot_rounds:
        individual_rounds, averaged_results, _ = load_individual_rounds_for_plotting(results_dir)
        # Convert to dataframes
        df_avg = pd.DataFrame(averaged_results)
        df_rounds = pd.DataFrame(individual_rounds)
        return df_avg, dataset_name, df_rounds
    else:
        individual_rounds, averaged_results, _ = load_individual_rounds_for_plotting(results_dir)
        df = pd.DataFrame(averaged_results)
        return df, dataset_name, None


def load_and_compare_data(inputs, plot_rounds=False):
    """ Load and merge multiple datasets for comparison.

    When plot_rounds=True, each round becomes a separate dataset.

    Args:
        inputs: List of paths to CSV files or result directories
        plot_rounds: If True, create separate datasets for each round

    Returns:
        Tuple of (merged_dataframe, dataset_names)
    """
    datasets = []
    dataset_names = []

    # Load all datasets
    for i, input_path in enumerate(inputs):
        input_path = Path(input_path)

        try:
            if is_results_directory(input_path):
                # Load from results directory
                if plot_rounds:
                    # Load individual rounds as separate datasets
                    individual_rounds, averaged_results, metadata = load_individual_rounds_for_plotting(input_path)

                    # Get base dataset name
                    experiment_meta = metadata.get('experiment_metadata', {})
                    model_name = experiment_meta.get('model_name', 'unknown')
                    text_name = experiment_meta.get('source_text_name', 'unknown')
                    model_id = experiment_meta.get('model_id', '')

                    # Strip organization from model name
                    if '/' in str(model_name):
                        model_name = str(model_name).split('/')[-1]

                    base_name_parts = [str(model_name), str(text_name), str(model_id)] if model_id else [str(model_name), str(text_name)]
                    base_name = ' '.join(base_name_parts)

                    # Convert individual rounds to dataframes, one per round number
                    rounds_by_number = {}
                    for round_data in individual_rounds:
                        round_num = round_data.get('round_number', 1)
                        if round_num not in rounds_by_number:
                            rounds_by_number[round_num] = []
                        rounds_by_number[round_num].append(round_data)

                    # Create a dataset for each round
                    for round_num in sorted(rounds_by_number.keys()):
                        round_df = pd.DataFrame(rounds_by_number[round_num])
                        round_df = round_df[round_df['context_length'].notna()].copy()
                        round_name = f"{base_name} R{round_num}"
                        datasets.append((round_df, round_name))
                        dataset_names.append(round_name)
                        print(f"Loaded round {round_num} ({len(round_df)} rows) from {input_path}")

                    # Also add averaged dataset
                    df_avg = pd.DataFrame(averaged_results)
                    df_avg = df_avg[df_avg['context_length'].notna()].copy()
                    avg_name = f"{base_name} AVG"
                    datasets.append((df_avg, avg_name))
                    dataset_names.append(avg_name)
                    print(f"Loaded average ({len(df_avg)} rows) from {input_path}")

                else:
                    # Load averaged results only
                    df, name, _ = load_data_from_directory(input_path, plot_rounds=False)
                    datasets.append((df, name))
                    dataset_names.append(name)
                    print(f"Loaded {len(df)} rows from {input_path}")
            else:
                # Load from CSV file
                df, name = load_csv_for_plotting(input_path)
                datasets.append((df, name))
                dataset_names.append(name)
                if plot_rounds:
                    print(f"Warning: Cannot plot individual rounds from CSV file: {input_path}")
                print(f"Loaded {len(df)} rows from {input_path}")

        except Exception as e:
            print(f"Error loading {input_path}: {e}")
            raise

    if not datasets:
        raise ValueError("No valid datasets loaded")

    # Check for duplicate dataset names (debugging)
    if len(dataset_names) != len(set(dataset_names)):
        print(f"Warning: Duplicate dataset names detected: {dataset_names}")
        print("Dataset names derived from CSV metadata.")

    # Start with the first dataset
    merged, first_name = datasets[0]
    merged = merged.copy()

    # Add suffix to columns (except context_length)
    columns_to_rename = [col for col in merged.columns if col != 'context_length']
    for col in columns_to_rename:
        merged = merged.rename(columns={col: f"{col}_{first_name}"})

    # Merge additional datasets
    for data, name in datasets[1:]:
        data_renamed = data.copy()
        # Add suffix to columns (except context_length)
        for col in columns_to_rename:
            if col in data_renamed.columns:
                data_renamed = data_renamed.rename(columns={col: f"{col}_{name}"})

        merged = pd.merge(merged, data_renamed, on='context_length', how='outer')

    return merged.sort_values('context_length'), dataset_names

def convert_enhanced_results_to_dataframe(enhanced_results, dataset_name):
    """ Convert enhanced_results list to pandas DataFrame.

    Args:
        enhanced_results: List of dictionaries from CSV writer
        dataset_name: Name for this dataset (deprecated, kept for compatibility)

    Returns:
        DataFrame with the data
    """
    df = pd.DataFrame(enhanced_results)
    # Filter out null context lengths
    df_clean = df[df['context_length'].notna()].copy()
    return df_clean

def get_plot_colors_and_markers():
    """ Get distinct colors and markers for up to 6 datasets.
    
    Returns:
        Tuple of (colors, markers) lists
    """
    colors = ['#2563eb', '#dc2626', '#16a34a', '#ca8a04', '#9333ea', '#c2410c']
    markers = ['o', 's', '^', 'D', 'v', '<']
    return colors, markers
    
def calculate_axis_ranges(data, dataset_names):
    """ Calculate appropriate axis ranges based on actual data for all four metrics.
    
    Args:
        data: Merged dataframe with all dataset metrics
        dataset_names: List of dataset names
        
    Returns:
        Dictionary with calculated ranges for each metric
    """
    ranges = {}
    
    # Vocabulary diversity
    vocab_values = []
    for dataset in dataset_names:
        col = f'vocabulary_diversity_{dataset}'
        if col in data.columns:
            vals = data[col].dropna()
            if len(vals) > 0:
                vocab_values.extend(vals)
    
    if vocab_values:
        vocab_min, vocab_max = min(vocab_values), max(vocab_values)
        vocab_range = vocab_max - vocab_min
        if vocab_min > 0.20:
            ranges['vocab_diversity'] = (0.20, 0.70)
        else:
            ranges['vocab_diversity'] = (vocab_min - .05, vocab_min + 0.45)
    else:
        ranges['vocab_diversity'] = (0.20, 0.70)
    
    # Cloze score
    cloze_values = []
    for dataset in dataset_names:
        col = f'cloze_score_{dataset}'
        if col in data.columns:
            vals = data[col].dropna()
            if len(vals) > 0:
                cloze_values.extend(vals)
    
    if cloze_values:
        cloze_min, cloze_max = min(cloze_values), max(cloze_values)
        cloze_range = cloze_max - cloze_min
        if cloze_min > 15:
            ranges['cloze_score'] = (15, 40)
        elif cloze_max > 35:
            ranges['cloze_score'] = (cloze_max - 24, cloze_max + 1) 
        else: 
            ranges['cloze_score'] = (cloze_min - 1, cloze_min + 24)
    else:
        ranges['cloze_score'] = (15, 40)
    
    # Adjacent coherence
    coherence_values = []
    for dataset in dataset_names:
        col = f'adjacent_coherence_{dataset}'
        if col in data.columns:
            vals = data[col].dropna()
            if len(vals) > 0:
                coherence_values.extend(vals)
    
    if coherence_values:
        coherence_min, coherence_max = min(coherence_values), max(coherence_values)
        if coherence_min > 0.4:
            ranges['adjacent_coherence'] = (0.4, 1.0)
        elif coherence_max < 0.6:
            ranges['adjacent_coherence'] = (0.0, 0.6)
        else:
            ranges['adjacent_coherence'] = (coherence_min - 0.1, coherence_min + 0.7)
        #coherence_range = coherence_max - coherence_min
        #coherence_padding = coherence_range * 0.1
        #ranges['adjacent_coherence'] = (coherence_min - coherence_padding, coherence_max + coherence_padding)
    else:
        ranges['adjacent_coherence'] = (0.0, 0.7)
    
    # Bigram repetition rate
    bigram_values = []
    for dataset in dataset_names:
        col = f'bigram_repetition_rate_{dataset}'
        if col in data.columns:
            vals = data[col].dropna()
            if len(vals) > 0:
                bigram_values.extend(vals)
    
    if bigram_values:
        bigram_min, bigram_max = min(bigram_values), max(bigram_values)
        if bigram_min > 0.2:
            ranges['bigram_repetition'] = (0.2, 0.5)
        elif bigram_max < 0.4:
            ranges['bigram_repetition'] = (0.0, 0.3)
        else:
            ranges['bigram_repetition'] = (bigram_min - 0.01, bigram_min + 0.31) 
        #bigram_range = bigram_max - bigram_min
        #bigram_padding = bigram_range * 0.1
        #ranges['bigram_repetition'] = (bigram_min - bigram_padding, bigram_max + bigram_padding)
    else:
        ranges['bigram_repetition'] = (0.0, 0.3)
    
    return ranges
    
def create_comparison_plots(data, dataset_names, output_file='comparison.png', dpi=300, silent=False):
    """ Create comprehensive performance comparison plots for multiple datasets.
    
    Creates four plots in a 2x2 layout:
    1. Vocabulary Diversity (creativity metric)
    2. Cloze Score (degradation metric)
    3. Adjacent Coherence (text quality metric)
    4. Bigram Repetition Rate (repetition metric)
    
    Args:
        data: Merged dataframe with all dataset metrics
        dataset_names: List of dataset names
        output_file: Path for output PNG file
        dpi: Resolution for output image
        silent: If True, suppress print statements
    """
    # Set backend to Agg for headless operation
    import matplotlib
    matplotlib.use('Agg')
    
    # Calculate dynamic ranges for all metrics
    ranges = calculate_axis_ranges(data, dataset_names)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Analysis', fontsize=16, fontweight='bold')
    
    colors, markers = get_plot_colors_and_markers()
    
    # Plot 1: Vocabulary Diversity (top-left)
    ax1 = axes[0, 0]

    plotted_any_data = False
    for j, dataset_name in enumerate(dataset_names):
        vocab_col = f'vocabulary_diversity_{dataset_name}'

        if vocab_col in data.columns:
            mask = data[vocab_col].notna() & data['context_length'].notna()
            if mask.any():
                values = data.loc[mask, vocab_col].tolist()
                ax1.plot(data.loc[mask, 'context_length'], values,
                        marker=markers[j % len(markers)], color=colors[j % len(colors)],
                        linewidth=3, markersize=6, label=dataset_name, linestyle='-')
                plotted_any_data = True
        else:
            if not silent:
                print(f"Warning: Column '{vocab_col}' not found in data")
    
    '''
    ax1.set_xscale('log')
    ax1.set_xlabel('Context Length')
    ax1.set_ylabel('Vocabulary Diversity')
    ax1.set_title('Vocabulary Diversity (Higher is More Diverse Wording)', fontweight='bold')
    if 'vocab_diversity' in ranges:
        ax1.set_ylim(ranges['vocab_diversity'])
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=8)
    '''
    ax1.set_xscale('log')
    ax1.set_xlabel('Context Length')
    ax1.set_ylabel('Vocabulary Diversity')
    ax1.set_title('Vocabulary Diversity (Higher is Less Diverse Wording)', fontweight='bold')
    if 'vocab_diversity' in ranges:
        ax1.set_ylim(ranges['vocab_diversity'])
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3)
    if plotted_any_data:
        ax1.legend(loc='best', fontsize=8)
    
    # Plot 2: Cloze Score (top-right)
    ax2 = axes[0, 1]

    has_cloze_data = False
    for j, dataset_name in enumerate(dataset_names):
        cloze_col = f'cloze_score_{dataset_name}'

        if cloze_col in data.columns:
            mask = data[cloze_col].notna() & data['context_length'].notna()
            if mask.any():
                values = data.loc[mask, cloze_col].tolist()
                ax2.plot(data.loc[mask, 'context_length'], values,
                        marker=markers[j % len(markers)], color=colors[j % len(colors)],
                        linewidth=3, markersize=6, label=dataset_name, linestyle='-')
                has_cloze_data = True

    ax2.set_xscale('log')
    ax2.set_xlabel('Context Length')
    ax2.set_ylabel('Cloze Score')
    ax2.set_title('Cloze Score (Higher is More Basic Wording)', fontweight='bold')
    if 'cloze_score' in ranges:
        ax2.set_ylim(ranges['cloze_score'])
    ax2.grid(True, alpha=0.3)
    if has_cloze_data:
        ax2.legend(loc='best', fontsize=8)
    
    # Plot 3: Adjacent Coherence (bottom-left)
    ax3 = axes[1, 0]

    has_coherence_data = False
    for j, dataset_name in enumerate(dataset_names):
        coherence_col = f'adjacent_coherence_{dataset_name}'

        if coherence_col in data.columns:
            mask = data[coherence_col].notna() & data['context_length'].notna()
            if mask.any():
                values = data.loc[mask, coherence_col].tolist()
                ax3.plot(data.loc[mask, 'context_length'], values,
                        marker=markers[j % len(markers)], color=colors[j % len(colors)],
                        linewidth=3, markersize=6, label=dataset_name, linestyle='-')
                has_coherence_data = True

    ax3.set_xscale('log')
    ax3.set_xlabel('Context Length')
    ax3.set_ylabel('Adjacent Similarity')
    ax3.set_title('Adjacent Sentence Similarity (Higher is More Similar)', fontweight='bold')
    if 'adjacent_coherence' in ranges:
        ax3.set_ylim(ranges['adjacent_coherence'])
    ax3.grid(True, alpha=0.3)
    if has_coherence_data:
        ax3.legend(loc='best', fontsize=8)
    else:
        ax3.text(0.5, 0.5, 'No adjacent_coherence data available',
                ha='center', va='center', transform=ax3.transAxes, fontsize=12, color='gray')
    
    # Plot 4: Bigram Repetition Rate (bottom-right)
    ax4 = axes[1, 1]

    has_bigram_data = False
    for j, dataset_name in enumerate(dataset_names):
        bigram_col = f'bigram_repetition_rate_{dataset_name}'

        if bigram_col in data.columns:
            mask = data[bigram_col].notna() & data['context_length'].notna()
            if mask.any():
                values = data.loc[mask, bigram_col].tolist()
                ax4.plot(data.loc[mask, 'context_length'], values,
                        marker=markers[j % len(markers)], color=colors[j % len(colors)],
                        linewidth=3, markersize=6, label=dataset_name, linestyle='-')
                has_bigram_data = True

    ax4.set_xscale('log')
    ax4.set_xlabel('Context Length')
    ax4.set_ylabel('Bigram Repetition Rate')
    ax4.set_title('Bigram Repetition Rate (Higher is More Repetitive)', fontweight='bold')
    if 'bigram_repetition' in ranges:
        ax4.set_ylim(ranges['bigram_repetition'])
    ax4.grid(True, alpha=0.3)
    if has_bigram_data:
        ax4.legend(loc='best', fontsize=8)
    
    # Format x-axis labels for all plots
    for ax in [ax1, ax2, ax3, ax4]:
        if len(data['context_length'].dropna()) > 0:
            unique_contexts = sorted(data['context_length'].dropna().unique())
            ax.set_xticks(unique_contexts)
            # Use 1024 for K (binary) instead of 1000 (decimal)
            ax.set_xticklabels([f'{int(x/1024)}K' if x >= 1024 else str(int(x))
                               for x in unique_contexts])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    if not silent:
        print(f"Plot saved to: {output_file}")
    plt.close()

def make_png(enhanced_results, output_file_path, silent=True):
    """ Generate PNG plots from enhanced_results data.
    
    Args:
        enhanced_results: List of dictionaries from CSV writer
        output_file_path: Path where CSV would be written (used to derive PNG names)
        silent: If True, suppress most output
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Convert enhanced_results to expected format
        output_path = Path(output_file_path)
        base_output_path = output_path.stem
        dataset_name = get_dataset_name(output_file_path)

        # Convert to DataFrame
        df = convert_enhanced_results_to_dataframe(enhanced_results, dataset_name)

        if len(df) == 0:
            if not silent:
                print("Error: No valid data found")
            return False

        # Generate output filenames in the same directory as the CSV
        output_dir = output_path.parent
        main_plot_file = output_dir / f"{base_output_path}.png"
        plot_file_with_error_bars = output_dir / f"{base_output_path}_stddev.png"
        
        # Create single-dataset merged format expected by plotting functions
        data = df.copy()
        dataset_names = [dataset_name]
        
        # Rename columns to match expected format
        columns_to_rename = [col for col in data.columns if col != 'context_length']
        for col in columns_to_rename:
            data = data.rename(columns={col: f"{col}_{dataset_name}"})
        
        # Create plots (convert Path to string for create_comparison_plots)
        create_comparison_plots(data, dataset_names, str(main_plot_file), silent=silent)
        
        return True
        
    except Exception as e:
        if not silent:
            print(f"Error creating plots: {e}")
        return False

def parse_arguments():
    """ Parse command line arguments for multiple CSV file inputs or result directories.

    Returns:
        Parsed arguments containing file paths and options
    """
    parser = argparse.ArgumentParser(
        description='Compare performance across multiple datasets and different context lengths',
        epilog='Supports 1-6 CSV files or result directories. Dataset names are derived from metadata.'
    )

    parser.add_argument('inputs', nargs='+',
                       help='Path(s) to CSV files or results directories (1-6 supported)')

    parser.add_argument('--plot-rounds', action='store_true',
                       help='Plot individual rounds in addition to averages (requires results directories)')

    return parser.parse_args()

def main():
    """ Main function to run the complete analysis.
    """
    args = parse_arguments()

    # Validate number of input files
    if len(args.inputs) > 6:
        print(f"Error: Too many inputs ({len(args.inputs)}). Maximum supported is 6.")
        return

    if len(args.inputs) < 1:
        print("Error: At least one input is required.")
        return

    try:
        # Load and merge data first to get dataset names
        print(f"Loading data from {len(args.inputs)} input(s)...")
        for i, f in enumerate(args.inputs, 1):
            input_type = "directory" if Path(f).is_dir() else "CSV file"
            print(f"  {i}. {f} ({input_type})")

        # Load data (with or without individual rounds as separate datasets)
        data, dataset_names = load_and_compare_data(args.inputs, plot_rounds=args.plot_rounds)
        print(f"Successfully merged {len(data)} context length comparisons")
        print(f"Datasets: {', '.join(dataset_names)}")

        # Generate output filename using dataset names
        main_output_file = generate_plot_filename(dataset_names)

        # Create plots (rounds are already separate datasets if plot_rounds=True)
        create_comparison_plots(data, dataset_names, main_output_file)
        
    except FileNotFoundError as e:
        print(f"Error: Could not find one or more CSV files.")
        print(f"Make sure all specified files exist and are accessible.")
        print(f"Details: {e}")
    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()