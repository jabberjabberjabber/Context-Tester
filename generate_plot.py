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

Requirements:
    pip install pandas matplotlib numpy
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from pathlib import Path

def get_dataset_name(filepath):
    """ Extract dataset name from file path (filename without extension).
    
    Args:
        filepath: Path to CSV file
    
    Returns:
        Clean dataset name for use in plots and analysis
    """
    return Path(filepath).stem

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

def get_plot_colors_and_markers():
    """ Get distinct colors and markers for up to 6 datasets.
    
    Returns:
        Tuple of (colors, markers) lists
    """
    colors = ['#2563eb', '#dc2626', '#16a34a', '#ca8a04', '#9333ea', '#c2410c']
    markers = ['o', 's', '^', 'D', 'v', '<']
    return colors, markers

def create_comparison_plots(data, dataset_names, output_file='comparison.png', dpi=300):
    """ Create comprehensive comparison plots for multiple datasets.
    
    Args:
        data: Merged dataframe with all dataset metrics
        dataset_names: List of dataset names
        output_file: Path for output PNG file
        dpi: Resolution for output image
    """
    # Set backend to Agg for headless operation
    import matplotlib
    matplotlib.use('Agg')
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Multi-Dataset Performance Comparison', fontsize=16, fontweight='bold')
    
    metrics = [
        ('cloze_score', 'Cloze Score', 'Score'),
        ('pct_unfamiliar_words', 'Unfamiliar Words', 'Percentage'),
        ('vocabulary_diversity', 'Vocabulary Diversity', 'Diversity Score'),
        ('continuation_length', 'Continuation Length', 'Characters'),
        ('avg_sentence_length', 'Average Sentence Length', 'Words'),
        ('sentence_length_variance', 'Sentence Length Variance', 'Variance')
    ]
    
    colors, markers = get_plot_colors_and_markers()
    
    for i, (metric, title, ylabel) in enumerate(metrics):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # Plot each dataset
        for j, dataset_name in enumerate(dataset_names):
            column_name = f'{metric}_{dataset_name}'
            if column_name in data.columns:
                # Convert percentage if needed
                values = data[column_name] * (100 if metric == 'pct_unfamiliar_words' else 1)
                # Filter out NaN values for plotting
                mask = values.notna() & data['context_length'].notna()
                if mask.any():
                    ax.plot(data.loc[mask, 'context_length'], values[mask], 
                           marker=markers[j % len(markers)], 
                           color=colors[j % len(colors)],
                           linewidth=3, markersize=6, label=dataset_name)
        
        # Formatting
        ax.set_xscale('log')
        ax.set_xlabel('Context Length')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Format x-axis labels
        if len(data['context_length'].dropna()) > 0:
            unique_contexts = sorted(data['context_length'].dropna().unique())
            ax.set_xticks(unique_contexts)
            ax.set_xticklabels([f'{int(x/1000)}K' if x >= 1000 else str(int(x)) 
                               for x in unique_contexts])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    plt.close()

def analyze_performance_ranges(data, dataset_names):
    """ Analyze performance across different context ranges for all datasets.
    
    Args:
        data: Merged dataframe with all dataset metrics
        dataset_names: List of dataset names
    """
    print("=" * 80)
    print("PERFORMANCE ANALYSIS BY CONTEXT LENGTH")
    print("=" * 80)
    
    # Get available context lengths
    available_contexts = sorted(data['context_length'].dropna().unique())
    
    for ctx in available_contexts:
        ctx_data = data[data['context_length'] == ctx]
        if len(ctx_data) == 0:
            continue
            
        print(f"\nContext Length: {int(ctx):,}")
        
        # Cloze score comparison
        print("  Cloze Scores:")
        cloze_scores = {}
        for dataset in dataset_names:
            col_name = f'cloze_score_{dataset}'
            if col_name in ctx_data.columns and not ctx_data[col_name].isna().all():
                score = ctx_data[col_name].iloc[0]
                cloze_scores[dataset] = score
                print(f"    {dataset}: {score:.3f}")
        
        if cloze_scores:
            best_cloze = max(cloze_scores, key=cloze_scores.get)
            print(f"    Winner: {best_cloze}")
        
        # Unfamiliar words (lower is better)
        print("  Unfamiliar Word Percentages:")
        unfam_scores = {}
        for dataset in dataset_names:
            col_name = f'pct_unfamiliar_words_{dataset}'
            if col_name in ctx_data.columns and not ctx_data[col_name].isna().all():
                score = ctx_data[col_name].iloc[0]
                unfam_scores[dataset] = score
                print(f"    {dataset}: {score*100:.1f}%")
        
        if unfam_scores:
            best_unfam = min(unfam_scores, key=unfam_scores.get)
            print(f"    Winner (lowest): {best_unfam}")
        
        # Vocabulary diversity
        print("  Vocabulary Diversity:")
        vocab_scores = {}
        for dataset in dataset_names:
            col_name = f'vocabulary_diversity_{dataset}'
            if col_name in ctx_data.columns and not ctx_data[col_name].isna().all():
                score = ctx_data[col_name].iloc[0]
                vocab_scores[dataset] = score
                print(f"    {dataset}: {score:.3f}")
        
        if vocab_scores:
            best_vocab = max(vocab_scores, key=vocab_scores.get)
            print(f"    Winner: {best_vocab}")

def create_summary_analysis(data, dataset_names):
    """ Create summary statistics and insights for all datasets.
    
    Args:
        data: Merged dataframe with all dataset metrics
        dataset_names: List of dataset names
    """
    print("\n" + "=" * 80)
    print("SUMMARY INSIGHTS")
    print("=" * 80)
    
    # Overall averages
    print("Overall Performance Averages:")
    
    metrics_to_analyze = ['cloze_score', 'vocabulary_diversity', 'pct_unfamiliar_words']
    
    for metric in metrics_to_analyze:
        print(f"\n  {metric.replace('_', ' ').title()}:")
        metric_values = {}
        
        for dataset in dataset_names:
            col_name = f'{metric}_{dataset}'
            if col_name in data.columns:
                avg_val = data[col_name].mean()
                if not pd.isna(avg_val):
                    metric_values[dataset] = avg_val
                    if metric == 'pct_unfamiliar_words':
                        print(f"    {dataset}: {avg_val*100:.1f}%")
                    elif metric == 'continuation_length':
                        print(f"    {dataset}: {avg_val:.0f} chars")
                        #pass
                    else:
                        print(f"    {dataset}: {avg_val:.3f}")
        
        # Find best performing dataset for this metric
        if metric_values:
            if metric == 'pct_unfamiliar_words':
                best_dataset = min(metric_values, key=metric_values.get)
                print(f"    Best (lowest): {best_dataset}")
            else:
                best_dataset = max(metric_values, key=metric_values.get)
                print(f"    Best: {best_dataset}")
    
    # Context-specific advantages
    print(f"\nPeak Performance by Context Length:")
    
    available_contexts = sorted(data['context_length'].dropna().unique())
    
    for dataset in dataset_names:
        cloze_col = f'cloze_score_{dataset}'
        if cloze_col in data.columns and not data[cloze_col].isna().all():
            best_idx = data[cloze_col].idxmax()
            if not pd.isna(best_idx):
                best_ctx = data.loc[best_idx, 'context_length']
                best_score = data.loc[best_idx, cloze_col]
                print(f"  {dataset}: {int(best_ctx):,} context (score: {best_score:.3f})")

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
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots (useful for headless environments)')
    parser.add_argument('--output', '-o', default='comparison.png',
                       help='Output PNG filename (default: comparison.png)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='Output image DPI (default: 300)')
    
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
        # Load and merge data
        print(f"Loading data from {len(args.csv_files)} file(s)...")
        for i, f in enumerate(args.csv_files, 1):
            print(f"  {i}. {f}")
        
        data, dataset_names = load_and_compare_data(args.csv_files)
        print(f"Successfully merged {len(data)} context length comparisons")
        print(f"Datasets: {', '.join(dataset_names)}")
        
        # Create visualizations (unless disabled)
        if not args.no_plots:
            print("Creating comparison plots...")
            create_comparison_plots(data, dataset_names, args.output, args.dpi)
        else:
            print("Skipping plots (--no-plots specified)")
        
        # Detailed analysis
        analyze_performance_ranges(data, dataset_names)
        
        # Summary insights
        create_summary_analysis(data, dataset_names)
        
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