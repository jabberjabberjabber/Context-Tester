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
    """ Create adaptive baseline comparison plots for multiple datasets.
    
    Creates two plots:
    1. Creativity Plot: vocab_diversity + sentence_length_variance
    2. Degradation Plot: cloze_score + adaptive sentence length penalty
    
    Args:
        data: Merged dataframe with all dataset metrics
        dataset_names: List of dataset names
        output_file: Path for output PNG file
        dpi: Resolution for output image
    """
    # Set backend to Agg for headless operation
    import matplotlib
    matplotlib.use('Agg')
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Model Performance Analysis: Creativity vs Degradation', fontsize=16, fontweight='bold')
    
    colors, markers = get_plot_colors_and_markers()
    
    # Fixed y-axis ranges for consistent cross-comparison
    VOCAB_DIVERSITY_RANGE = (0.20, 0.70)
    SENTENCE_VARIANCE_RANGE = (20, 200)
    CLOZE_SCORE_RANGE = (15, 40)
    SENTENCE_PENALTY_RANGE = (0, 2.0)  # Log penalty range
    
    # Plot 1: Creativity Metrics
    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    
    for j, dataset_name in enumerate(dataset_names):
        vocab_col = f'vocabulary_diversity_{dataset_name}'
        variance_col = f'sentence_length_variance_{dataset_name}'
        
        if vocab_col in data.columns:
            mask = data[vocab_col].notna() & data['context_length'].notna()
            if mask.any():
                ax1.plot(data.loc[mask, 'context_length'], data.loc[mask, vocab_col], 
                        marker=markers[j % len(markers)], color=colors[j % len(colors)],
                        linewidth=3, markersize=6, label=f'{dataset_name} (vocab)', linestyle='-')
        
        if variance_col in data.columns:
            mask = data[variance_col].notna() & data['context_length'].notna()
            if mask.any():
                ax1_twin.plot(data.loc[mask, 'context_length'], data.loc[mask, variance_col], 
                             marker=markers[j % len(markers)], color=colors[j % len(colors)],
                             linewidth=3, markersize=6, label=f'{dataset_name} (variance)', linestyle='--', alpha=0.7)
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Context Length')
    ax1.set_ylabel('Vocabulary Diversity', color='black')
    ax1.set_title('Creativity Metrics', fontweight='bold')
    ax1.set_ylim(VOCAB_DIVERSITY_RANGE)
    ax1.grid(True, alpha=0.3)
    
    ax1_twin.set_ylabel('Sentence Length Variance', color='gray')
    ax1_twin.set_ylim(SENTENCE_VARIANCE_RANGE)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    # Plot 2: Degradation Metrics (with adaptive baseline)
    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    
    for j, dataset_name in enumerate(dataset_names):
        cloze_col = f'cloze_score_{dataset_name}'
        sentence_col = f'avg_sentence_length_{dataset_name}'
        
        # Plot cloze score
        if cloze_col in data.columns:
            mask = data[cloze_col].notna() & data['context_length'].notna()
            if mask.any():
                ax2.plot(data.loc[mask, 'context_length'], data.loc[mask, cloze_col], 
                        marker=markers[j % len(markers)], color=colors[j % len(colors)],
                        linewidth=3, markersize=6, label=f'{dataset_name} (cloze)', linestyle='-')
        
        # Calculate adaptive sentence penalty
        if sentence_col in data.columns and cloze_col in data.columns:
            # Find optimal point (minimum cloze score)
            dataset_data = data[[cloze_col, sentence_col, 'context_length']].dropna()
            if len(dataset_data) > 0:
                optimal_idx = dataset_data[cloze_col].idxmin()
                baseline_sentence_length = dataset_data.loc[optimal_idx, sentence_col]
                
                # Calculate log penalty for deviations from baseline
                sentence_penalty = np.log(np.maximum(baseline_sentence_length / dataset_data[sentence_col], 0.1))
                sentence_penalty = np.clip(sentence_penalty, 0, SENTENCE_PENALTY_RANGE[1])
                
                ax2_twin.plot(dataset_data['context_length'], sentence_penalty, 
                             marker=markers[j % len(markers)], color=colors[j % len(colors)],
                             linewidth=3, markersize=6, label=f'{dataset_name} (sent penalty)', 
                             linestyle='--', alpha=0.7)
    
    ax2.set_xscale('log')
    ax2.set_xlabel('Context Length')
    ax2.set_ylabel('Cloze Score (Predictability)', color='black')
    ax2.set_title('Degradation Metrics', fontweight='bold')
    ax2.set_ylim(CLOZE_SCORE_RANGE)
    ax2.grid(True, alpha=0.3)
    
    ax2_twin.set_ylabel('Sentence Length Penalty (Log)', color='gray')
    ax2_twin.set_ylim(SENTENCE_PENALTY_RANGE)
    
    # Combined legend
    lines3, labels3 = ax2.get_legend_handles_labels()
    lines4, labels4 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines3 + lines4, labels3 + labels4, loc='best')
    
    # Format x-axis labels for both plots
    for ax in [ax1, ax2]:
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
    """ Analyze creativity and degradation metrics across context ranges.
    
    Args:
        data: Merged dataframe with all dataset metrics
        dataset_names: List of dataset names
    """
    print("=" * 80)
    print("CREATIVITY & DEGRADATION ANALYSIS BY CONTEXT LENGTH")
    print("=" * 80)
    
    # Get available context lengths
    available_contexts = sorted(data['context_length'].dropna().unique())
    
    for ctx in available_contexts:
        ctx_data = data[data['context_length'] == ctx]
        if len(ctx_data) == 0:
            continue
            
        print(f"\nContext Length: {int(ctx):,}")
        
        # Creativity Metrics
        print("  CREATIVITY METRICS:")
        
        # Vocabulary diversity (higher = better)
        print("    Vocabulary Diversity:")
        vocab_scores = {}
        for dataset in dataset_names:
            col_name = f'vocabulary_diversity_{dataset}'
            if col_name in ctx_data.columns and not ctx_data[col_name].isna().all():
                score = ctx_data[col_name].iloc[0]
                vocab_scores[dataset] = score
                print(f"      {dataset}: {score:.3f}")
        
        if vocab_scores:
            best_vocab = max(vocab_scores, key=vocab_scores.get)
            print(f"      Winner: {best_vocab}")
        
        # Sentence length variance (higher = better)
        print("    Sentence Length Variance:")
        variance_scores = {}
        for dataset in dataset_names:
            col_name = f'sentence_length_variance_{dataset}'
            if col_name in ctx_data.columns and not ctx_data[col_name].isna().all():
                score = ctx_data[col_name].iloc[0]
                variance_scores[dataset] = score
                print(f"      {dataset}: {score:.1f}")
        
        if variance_scores:
            best_variance = max(variance_scores, key=variance_scores.get)
            print(f"      Winner: {best_variance}")
        
        # Degradation Metrics
        print("  DEGRADATION METRICS:")
        
        # Cloze score (lower = better, more sophisticated)
        print("    Cloze Scores (lower = better):")
        cloze_scores = {}
        for dataset in dataset_names:
            col_name = f'cloze_score_{dataset}'
            if col_name in ctx_data.columns and not ctx_data[col_name].isna().all():
                score = ctx_data[col_name].iloc[0]
                cloze_scores[dataset] = score
                print(f"      {dataset}: {score:.3f}")
        
        if cloze_scores:
            best_cloze = min(cloze_scores, key=cloze_scores.get)
            print(f"      Winner (lowest): {best_cloze}")
        
        # Average sentence length consistency
        print("    Average Sentence Length:")
        sentence_scores = {}
        for dataset in dataset_names:
            col_name = f'avg_sentence_length_{dataset}'
            if col_name in ctx_data.columns and not ctx_data[col_name].isna().all():
                score = ctx_data[col_name].iloc[0]
                sentence_scores[dataset] = score
                print(f"      {dataset}: {score:.1f} words")
        
        # Show deviation from each model's optimal baseline
        print("    Sentence Length Penalties (from each model's optimum):")
        for dataset in dataset_names:
            cloze_col = f'cloze_score_{dataset}'
            sentence_col = f'avg_sentence_length_{dataset}'
            
            if cloze_col in data.columns and sentence_col in data.columns:
                dataset_data = data[[cloze_col, sentence_col, 'context_length']].dropna()
                if len(dataset_data) > 0:
                    optimal_idx = dataset_data[cloze_col].idxmin()
                    baseline_sentence_length = dataset_data.loc[optimal_idx, sentence_col]
                    current_sentence_length = ctx_data[sentence_col].iloc[0] if not ctx_data[sentence_col].isna().all() else None
                    
                    if current_sentence_length is not None:
                        penalty = np.log(max(baseline_sentence_length / current_sentence_length, 0.1))
                        penalty = max(0, min(penalty, 2.0))  # Clip to reasonable range
                        print(f"      {dataset}: {penalty:.3f} (baseline: {baseline_sentence_length:.1f})")


def create_summary_analysis(data, dataset_names):
    """ Create summary statistics focusing on creativity and degradation patterns.
    
    Args:
        data: Merged dataframe with all dataset metrics
        dataset_names: List of dataset names
    """
    print("\n" + "=" * 80)
    print("SUMMARY INSIGHTS: CREATIVITY vs DEGRADATION")
    print("=" * 80)
    
    # Overall creativity averages
    print("CREATIVITY PERFORMANCE AVERAGES:")
    
    creativity_metrics = ['vocabulary_diversity', 'sentence_length_variance']
    
    for metric in creativity_metrics:
        print(f"\n  {metric.replace('_', ' ').title()}:")
        metric_values = {}
        
        for dataset in dataset_names:
            col_name = f'{metric}_{dataset}'
            if col_name in data.columns:
                avg_val = data[col_name].mean()
                if not pd.isna(avg_val):
                    metric_values[dataset] = avg_val
                    print(f"    {dataset}: {avg_val:.3f}")
        
        # Find best performing dataset for this metric
        if metric_values:
            best_dataset = max(metric_values, key=metric_values.get)
            print(f"    Best: {best_dataset}")
    
    # Degradation analysis
    print(f"\nDEGRADATION ANALYSIS:")
    
    print("  Average Cloze Scores (lower = better):")
    for dataset in dataset_names:
        cloze_col = f'cloze_score_{dataset}'
        if cloze_col in data.columns:
            avg_val = data[cloze_col].mean()
            if not pd.isna(avg_val):
                print(f"    {dataset}: {avg_val:.3f}")
    
    # Find each model's optimal context length and performance
    print(f"\nOPTIMAL PERFORMANCE POINTS:")
    
    for dataset in dataset_names:
        cloze_col = f'cloze_score_{dataset}'
        vocab_col = f'vocabulary_diversity_{dataset}'
        variance_col = f'sentence_length_variance_{dataset}'
        
        if cloze_col in data.columns and not data[cloze_col].isna().all():
            # Find the optimal point (minimum cloze score)
            dataset_data = data[[cloze_col, 'context_length', vocab_col, variance_col]].dropna()
            if len(dataset_data) > 0:
                optimal_idx = dataset_data[cloze_col].idxmin()
                optimal_ctx = dataset_data.loc[optimal_idx, 'context_length']
                optimal_cloze = dataset_data.loc[optimal_idx, cloze_col]
                optimal_vocab = dataset_data.loc[optimal_idx, vocab_col] if vocab_col in dataset_data.columns else None
                optimal_variance = dataset_data.loc[optimal_idx, variance_col] if variance_col in dataset_data.columns else None
                
                print(f"  {dataset}:")
                print(f"    Optimal context: {int(optimal_ctx):,}")
                print(f"    Cloze score: {optimal_cloze:.3f}")
                if optimal_vocab is not None:
                    print(f"    Vocab diversity: {optimal_vocab:.3f}")
                if optimal_variance is not None:
                    print(f"    Sentence variance: {optimal_variance:.1f}")
    
    # Degradation patterns
    print(f"\nDEGRADATION PATTERNS:")
    available_contexts = sorted(data['context_length'].dropna().unique())
    
    if len(available_contexts) >= 3:
        early_ctx = available_contexts[1]  # Skip 1K, use 2K
        late_ctx = available_contexts[-1]  # Use maximum context
        
        print(f"  Performance change from {int(early_ctx/1000)}K to {int(late_ctx/1000)}K context:")
        
        for dataset in dataset_names:
            cloze_col = f'cloze_score_{dataset}'
            vocab_col = f'vocabulary_diversity_{dataset}'
            
            if cloze_col in data.columns and vocab_col in data.columns:
                early_data = data[data['context_length'] == early_ctx]
                late_data = data[data['context_length'] == late_ctx]
                
                if len(early_data) > 0 and len(late_data) > 0:
                    early_cloze = early_data[cloze_col].iloc[0] if not early_data[cloze_col].isna().all() else None
                    late_cloze = late_data[cloze_col].iloc[0] if not late_data[cloze_col].isna().all() else None
                    early_vocab = early_data[vocab_col].iloc[0] if not early_data[vocab_col].isna().all() else None
                    late_vocab = late_data[vocab_col].iloc[0] if not late_data[vocab_col].isna().all() else None
                    
                    if all(v is not None for v in [early_cloze, late_cloze, early_vocab, late_vocab]):
                        cloze_change = late_cloze - early_cloze
                        vocab_change = late_vocab - early_vocab
                        
                        print(f"    {dataset}:")
                        print(f"      Cloze change: {cloze_change:+.3f} ({'worse' if cloze_change > 0 else 'better'})")
                        print(f"      Vocab change: {vocab_change:+.3f} ({'worse' if vocab_change < 0 else 'better'})")


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