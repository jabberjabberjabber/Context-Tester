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

def get_dataset_name(filepath):
    """ Extract dataset name from file path (filename without extension).
    
    Args:
        filepath: Path to CSV file
    
    Returns:
        Clean dataset name for use in plots and analysis
    """
    return Path(filepath).stem

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

def generate_output_filename(csv_files):
    """ Generate output PNG filename based on input CSV files.
    
    Args:
        csv_files: List of CSV file paths
    
    Returns:
        Generated PNG filename
    """
    if len(csv_files) == 1:
        return Path(csv_files[0]).stem + '.png'
    
    # Multiple files: first_base_with-other1-other2.png
    first_base = get_base_name(csv_files[0])
    other_bases = [get_base_name(f) for f in csv_files[1:]]
    
    return f"{first_base}_with-{'-'.join(other_bases)}.png"

def load_and_compare_data(csv_files):
    """ Load and merge multiple datasets for comparison.
    
    Args:
        csv_files: List of paths to CSV files
    
    Returns:
        Tuple of (merged_dataframe, dataset_names)
    """
    dataset_names = [get_dataset_name(f) for f in csv_files]
    datasets = []
    
    # Load all datasets
    for i, csv_file in enumerate(csv_files):
        try:
            data = pd.read_csv(csv_file)
            # Filter out null context lengths
            data_clean = data[data['context_length'].notna()].copy()
            datasets.append((data_clean, dataset_names[i]))
            print(f"Loaded {len(data_clean)} rows from {csv_file}")
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            raise
    
    if not datasets:
        raise ValueError("No valid datasets loaded")
    
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
        dataset_name: Name for this dataset
        
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
    
    for j, dataset_name in enumerate(dataset_names):
        vocab_col = f'vocabulary_diversity_{dataset_name}'
        
        if vocab_col in data.columns:
            mask = data[vocab_col].notna() & data['context_length'].notna()
            if mask.any():
                values = data.loc[mask, vocab_col].tolist()
                ax1.plot(data.loc[mask, 'context_length'], values, 
                        marker=markers[j % len(markers)], color=colors[j % len(colors)],
                        linewidth=3, markersize=6, label=dataset_name, linestyle='-')
    
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
    ax1.legend(loc='best', fontsize=8)
    
    # Plot 2: Cloze Score (top-right)
    ax2 = axes[0, 1]
    
    for j, dataset_name in enumerate(dataset_names):
        cloze_col = f'cloze_score_{dataset_name}'
        
        if cloze_col in data.columns:
            mask = data[cloze_col].notna() & data['context_length'].notna()
            if mask.any():
                values = data.loc[mask, cloze_col].tolist()
                ax2.plot(data.loc[mask, 'context_length'], values, 
                        marker=markers[j % len(markers)], color=colors[j % len(colors)],
                        linewidth=3, markersize=6, label=dataset_name, linestyle='-')
    
    ax2.set_xscale('log')
    ax2.set_xlabel('Context Length')
    ax2.set_ylabel('Cloze Score')
    ax2.set_title('Cloze Score (Higher is More Basic Wording)', fontweight='bold')
    if 'cloze_score' in ranges:
        ax2.set_ylim(ranges['cloze_score'])
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=8)
    
    # Plot 3: Adjacent Coherence (bottom-left)
    ax3 = axes[1, 0]
    
    for j, dataset_name in enumerate(dataset_names):
        coherence_col = f'adjacent_coherence_{dataset_name}'
        
        if coherence_col in data.columns:
            mask = data[coherence_col].notna() & data['context_length'].notna()
            if mask.any():
                values = data.loc[mask, coherence_col].tolist()
                ax3.plot(data.loc[mask, 'context_length'], values, 
                        marker=markers[j % len(markers)], color=colors[j % len(colors)],
                        linewidth=3, markersize=6, label=dataset_name, linestyle='-')
    
    ax3.set_xscale('log')
    ax3.set_xlabel('Context Length')
    ax3.set_ylabel('Adjacent Similarity')
    ax3.set_title('Adjacent Sentence Similarity (Higher is More Similar)', fontweight='bold')
    if 'adjacent_coherence' in ranges:
        ax3.set_ylim(ranges['adjacent_coherence'])
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best', fontsize=8)
    
    # Plot 4: Bigram Repetition Rate (bottom-right)
    ax4 = axes[1, 1]
    
    for j, dataset_name in enumerate(dataset_names):
        bigram_col = f'bigram_repetition_rate_{dataset_name}'
        
        if bigram_col in data.columns:
            mask = data[bigram_col].notna() & data['context_length'].notna()
            if mask.any():
                values = data.loc[mask, bigram_col].tolist()
                ax4.plot(data.loc[mask, 'context_length'], values, 
                        marker=markers[j % len(markers)], color=colors[j % len(colors)],
                        linewidth=3, markersize=6, label=dataset_name, linestyle='-')
    
    ax4.set_xscale('log')
    ax4.set_xlabel('Context Length')
    ax4.set_ylabel('Bigram Repetition Rate')
    ax4.set_title('Bigram Repetition Rate (Higher is More Repetitive)', fontweight='bold')
    if 'bigram_repetition' in ranges:
        ax4.set_ylim(ranges['bigram_repetition'])
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='best', fontsize=8)
    
    # Format x-axis labels for all plots
    for ax in [ax1, ax2, ax3, ax4]:
        if len(data['context_length'].dropna()) > 0:
            unique_contexts = sorted(data['context_length'].dropna().unique())
            ax.set_xticks(unique_contexts)
            ax.set_xticklabels([f'{int(x/1000)}K' if x >= 1000 else str(int(x)) 
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
        base_output_path = Path(output_file_path).stem
        dataset_name = get_dataset_name(output_file_path)
        
        # Convert to DataFrame
        df = convert_enhanced_results_to_dataframe(enhanced_results, dataset_name)
        
        if len(df) == 0:
            if not silent:
                print("Error: No valid data found")
            return False
        
        # Generate output filenames
        main_plot_file = f"{base_output_path}.png"
        plot_file_with_error_bars = f"{base_output_path}_stddev.png"
        
        # Create single-dataset merged format expected by plotting functions
        data = df.copy()
        dataset_names = [dataset_name]
        
        # Rename columns to match expected format
        columns_to_rename = [col for col in data.columns if col != 'context_length']
        for col in columns_to_rename:
            data = data.rename(columns={col: f"{col}_{dataset_name}"})
        
        # Create plots
        create_comparison_plots(data, dataset_names, main_plot_file, silent=silent)
        
        return True
        
    except Exception as e:
        if not silent:
            print(f"Error creating plots: {e}")
        return False

def parse_arguments():
    """ Parse command line arguments for multiple CSV file inputs.
    
    Returns:
        Parsed arguments containing file paths and options
    """
    parser = argparse.ArgumentParser(
        description='Compare performance across multiple datasets and different context lengths',
        epilog='Supports 1-6 CSV input files. Dataset names are derived from filenames.'
    )
    
    parser.add_argument('csv_files', nargs='+', 
                       help='Path(s) to CSV files (1-6 files supported)')

    return parser.parse_args()

def main():
    """ Main function to run the complete analysis.
    """
    args = parse_arguments()
    
    # Validate number of input files
    if len(args.csv_files) > 6:
        print(f"Error: Too many input files ({len(args.csv_files)}). Maximum supported is 6.")
        return
    
    if len(args.csv_files) < 1:
        print("Error: At least one CSV file is required.")
        return
    
    try:
        # Generate output filename automatically
        main_output_file = generate_output_filename(args.csv_files)
        
        # Load and merge data
        print(f"Loading data from {len(args.csv_files)} file(s)...")
        for i, f in enumerate(args.csv_files, 1):
            print(f"  {i}. {f}")
        
        data, dataset_names = load_and_compare_data(args.csv_files)
        print(f"Successfully merged {len(data)} context length comparisons")
        print(f"Datasets: {', '.join(dataset_names)}")
        
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