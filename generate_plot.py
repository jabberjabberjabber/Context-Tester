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
    """ Calculate appropriate axis ranges based on actual data.
    
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
        # Check lowest point on Y axis
        if vocab_min > 0.20:
            # Use the common Y values if it fits
            ranges['vocab_diversity'] = (0.20, 0.70)
        # Otherwise move the Y axis up while keeping the same range
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
        cloze_padding = cloze_range * 0.1
        # Check lowest point on Y axis
        if cloze_min > 15:
            # Use the common Y values if it fits
            ranges['cloze_score'] = (15, 40)
        # Otherwise move the Y axis up while keeping the same range
        else: 
            ranges['cloze_score'] = (cloze_min - 2, cloze_min + 23)
    else:
        ranges['cloze_score'] = (15, 40)
    
    return ranges

def align_twin_axis(primary_range, secondary_range, primary_values, secondary_values):
    """ Calculate aligned secondary axis range to match primary axis positioning.
    
    Args:
        primary_range: (min, max) for primary axis
        secondary_range: (min, max) for secondary axis  
        primary_values: List of primary axis values
        secondary_values: List of secondary axis values
        
    Returns:
        Adjusted secondary range (min, max)
    """
    if not primary_values or not secondary_values:
        return secondary_range
    
    # Find middle points of actual data
    primary_mid = (min(primary_values) + max(primary_values)) / 2
    secondary_mid = (min(secondary_values) + max(secondary_values)) / 2
    
    # Calculate where the primary middle falls in its range (0-1)
    primary_span = primary_range[1] - primary_range[0]
    primary_position = (primary_mid - primary_range[0]) / primary_span if primary_span > 0 else 0.5
    
    # Calculate secondary range to position its middle at same relative position
    secondary_span = secondary_range[1] - secondary_range[0]
    target_secondary_min = secondary_mid - (primary_position * secondary_span)
    target_secondary_max = target_secondary_min + secondary_span
    
    return (target_secondary_min, target_secondary_max)

def create_comparison_plots(data, dataset_names, output_file='comparison.png', dpi=300, silent=False):
    """ Create adaptive baseline comparison plots for multiple datasets.
    
    Creates two plots:
    1. Creativity Plot: vocab_diversity + sentence_length_variance
    2. Degradation Plot: cloze_score + adaptive sentence length penalty
    
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
    
    # Calculate dynamic ranges
    ranges = calculate_axis_ranges(data, dataset_names)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Model Performance Analysis: Creativity vs Degradation', fontsize=16, fontweight='bold')
    
    colors, markers = get_plot_colors_and_markers()
    
    # Plot 1: Creativity Metrics
    ax1 = axes[0]
    
    # Collect values for axis alignment
    vocab_plot_values = []
    variance_plot_values = []
    
    for j, dataset_name in enumerate(dataset_names):
        vocab_col = f'vocabulary_diversity_{dataset_name}'
        
        if vocab_col in data.columns:
            mask = data[vocab_col].notna() & data['context_length'].notna()
            if mask.any():
                values = data.loc[mask, vocab_col].tolist()
                vocab_plot_values.extend(values)
                ax1.plot(data.loc[mask, 'context_length'], values, 
                        marker=markers[j % len(markers)], color=colors[j % len(colors)],
                        linewidth=3, markersize=6, label=f'{dataset_name} (vocab)', linestyle='-')
        
     
    
    # Set primary axis
    ax1.set_xscale('log')
    ax1.set_xlabel('Context Length')
    ax1.set_ylabel('Vocabulary Diversity', color='black')
    ax1.set_title('Creativity Metrics', fontweight='bold')
    ax1.set_ylim(ranges['vocab_diversity'])
    ax1.grid(True, alpha=0.3)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc='best')
    
    # Plot 2: Degradation Metrics (with adaptive baseline)
    ax2 = axes[1]
    
    # Collect values for axis alignment
    cloze_plot_values = []
    penalty_plot_values = []
    
    for j, dataset_name in enumerate(dataset_names):
        cloze_col = f'cloze_score_{dataset_name}'
        
        # Plot cloze score
        if cloze_col in data.columns:
            mask = data[cloze_col].notna() & data['context_length'].notna()
            if mask.any():
                values = data.loc[mask, cloze_col].tolist()
                cloze_plot_values.extend(values)
                ax2.plot(data.loc[mask, 'context_length'], values, 
                        marker=markers[j % len(markers)], color=colors[j % len(colors)],
                        linewidth=3, markersize=6, label=f'{dataset_name} (cloze)', linestyle='-')
        
        
    # Set primary axis
    ax2.set_xscale('log')
    ax2.set_xlabel('Context Length')
    ax2.set_ylabel('Cloze Score (Predictability)', color='black')
    ax2.set_title('Degradation Metrics', fontweight='bold')
    ax2.set_ylim(ranges['cloze_score'])
    ax2.grid(True, alpha=0.3)
    
    # Combined legend
    lines3, labels3 = ax2.get_legend_handles_labels()
    ax2.legend(lines3, labels3, loc='best')
    
    # Format x-axis labels for both plots
    for ax in [ax1, ax2]:
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

def create_detailed_metrics_plots(data, dataset_names, output_file='detailed_metrics.png', dpi=300, silent=False):
    """ Create detailed metrics plots for additional analysis.
    
    Creates four plots:
    a) continuation_length (log scale)
    b) density metrics (linear scale) 
    c) ratio metrics (log scale 0-1)
    d) avg_word_length (linear scale)
    
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
    
    fig, axes = plt.subplots(2, 2, figsize=(17.5, 14))
    fig.suptitle('Detailed Metrics Analysis', fontsize=16, fontweight='bold', y=0.95)
    
    colors, markers = get_plot_colors_and_markers()
    
    # Plot A: Continuation Length (log scale)
    ax_a = axes[0, 0]
    continuation_values = []
    
    for j, dataset_name in enumerate(dataset_names):
        cont_col = f'continuation_length_{dataset_name}'
        if cont_col in data.columns:
            mask = data[cont_col].notna() & data['context_length'].notna()
            if mask.any():
                values = data.loc[mask, cont_col]
                continuation_values.extend(values.tolist())
                ax_a.plot(data.loc[mask, 'context_length'], values,
                         marker=markers[j % len(markers)], color=colors[j % len(colors)],
                         linewidth=3, markersize=6, label=dataset_name)
    
    ax_a.set_xscale('log')
    ax_a.set_yscale('log')
    ax_a.set_xlabel('Context Length')
    ax_a.set_ylabel('Continuation Length (log scale)')
    ax_a.set_title('Continuation Length')
    ax_a.grid(True, alpha=0.3)
    ax_a.legend(loc='best', fontsize=8)
    
    # Set y-range for continuation length
    if continuation_values:
        min_cont, max_cont = min(continuation_values), max(continuation_values)
        ax_a.set_ylim(max(1, min_cont * 0.8), max_cont * 1.2)
    
    # Plot B: Density Metrics (linear scale)
    ax_b = axes[0, 1]
    density_metrics = ['comma_density', 'semicolon_density', 'question_density', 'exclamation_density']
    all_density_values = []
    
    for metric in density_metrics:
        for j, dataset_name in enumerate(dataset_names):
            col = f'{metric}_{dataset_name}'
            if col in data.columns:
                mask = data[col].notna() & data['context_length'].notna()
                if mask.any():
                    values = data.loc[mask, col]
                    all_density_values.extend(values.tolist())
                    linestyle = ['-', '--', '-.', ':'][density_metrics.index(metric)]
                    ax_b.plot(data.loc[mask, 'context_length'], values,
                             marker=markers[j % len(markers)], color=colors[j % len(colors)],
                             linewidth=2, markersize=4, 
                             label=f'{dataset_name} ({metric.replace("_density", "")})',
                             linestyle=linestyle)
    
    ax_b.set_xscale('log')
    ax_b.set_xlabel('Context Length')
    ax_b.set_ylabel('Density')
    ax_b.set_title('Punctuation Densities')
    ax_b.grid(True, alpha=0.3)
    
    # Set y-range for density
    if all_density_values:
        max_density = max(all_density_values)
        ax_b.set_ylim(0, max_density * 1.1)
    
    # Plot C: Ratio Metrics (log scale 0-1)
    ax_c = axes[1, 0]
    ratio_metrics = ['function_word_ratio', 'unique_word_ratio_100', 'long_word_ratio']
    
    for metric in ratio_metrics:
        for j, dataset_name in enumerate(dataset_names):
            col = f'{metric}_{dataset_name}'
            if col in data.columns:
                mask = data[col].notna() & data['context_length'].notna()
                if mask.any():
                    values = data.loc[mask, col]
                    linestyle = ['-', '--', '-.'][ratio_metrics.index(metric)]
                    ax_c.plot(data.loc[mask, 'context_length'], values,
                             marker=markers[j % len(markers)], color=colors[j % len(colors)],
                             linewidth=2, markersize=4,
                             label=f'{dataset_name} ({metric.replace("_ratio", "").replace("_", " ")})',
                             linestyle=linestyle)
    
    ax_c.set_xscale('log')
    ax_c.set_yscale('log')
    ax_c.set_xlabel('Context Length')
    ax_c.set_ylabel('Ratio (log scale)')
    ax_c.set_title('Word Ratios')
    ax_c.set_ylim(0.001, 1)
    ax_c.grid(True, alpha=0.3)
    
    # Plot D: Average Word Length (linear scale)
    ax_d = axes[1, 1]
    word_length_values = []
    
    for j, dataset_name in enumerate(dataset_names):
        wl_col = f'avg_word_length_{dataset_name}'
        if wl_col in data.columns:
            mask = data[wl_col].notna() & data['context_length'].notna()
            if mask.any():
                values = data.loc[mask, wl_col]
                word_length_values.extend(values.tolist())
                ax_d.plot(data.loc[mask, 'context_length'], values,
                         marker=markers[j % len(markers)], color=colors[j % len(colors)],
                         linewidth=3, markersize=6, label=dataset_name)
    
    ax_d.set_xscale('log')
    ax_d.set_xlabel('Context Length')
    ax_d.set_ylabel('Average Word Length')
    ax_d.set_title('Average Word Length')
    ax_d.grid(True, alpha=0.3)
    ax_d.legend(loc='best', fontsize=8)
    
    # Set y-range for word length
    if word_length_values:
        min_wl, max_wl = min(word_length_values), max(word_length_values)
        wl_range = max_wl - min_wl
        padding = wl_range * 0.1
        ax_d.set_ylim(min_wl - padding, max_wl + padding)
    
    # Format x-axis labels for all plots
    for ax in [ax_a, ax_b, ax_c, ax_d]:
        if len(data['context_length'].dropna()) > 0:
            unique_contexts = sorted(data['context_length'].dropna().unique())
            ax.set_xticks(unique_contexts)
            ax.set_xticklabels([f'{int(x/1000)}K' if x >= 1000 else str(int(x)) 
                               for x in unique_contexts])
    
    # Add legends for plots B and C (which have multiple series per dataset)
    ax_b.legend(loc='upper right', fontsize=8)
    ax_c.legend(loc='upper right', fontsize=8)
    
    # Set equal aspect ratio for all subplots to make them more square
    for ax in [ax_a, ax_b, ax_c, ax_d]:
        ax.set_aspect('auto')
    
    # Adjust layout with proper spacing
    plt.subplots_adjust(left=0.06, bottom=0.08, right=0.95, top=0.90, wspace=0.25, hspace=0.3)
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    #if not silent:
        #print(f"Detailed metrics plot saved to: {output_file}")
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
    
    creativity_metrics = ['vocabulary_diversity']
    
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
            dataset_data = data[[cloze_col, 'context_length', vocab_col]].dropna()
            if len(dataset_data) > 0:
                optimal_idx = dataset_data[cloze_col].idxmin()
                optimal_ctx = dataset_data.loc[optimal_idx, 'context_length']
                optimal_cloze = dataset_data.loc[optimal_idx, cloze_col]
                optimal_vocab = dataset_data.loc[optimal_idx, vocab_col] if vocab_col in dataset_data.columns else None
                
                print(f"  {dataset}:")
                print(f"    Optimal context: {int(optimal_ctx):,}")
                print(f"    Cloze score: {optimal_cloze:.3f}")
                if optimal_vocab is not None:
                    print(f"    Vocab diversity: {optimal_vocab:.3f}")

    
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
        #detailed_plot_file = f"{base_output_path}_detailed.png"
        
        # Create single-dataset merged format expected by plotting functions
        data = df.copy()
        dataset_names = [dataset_name]
        
        # Rename columns to match expected format
        columns_to_rename = [col for col in data.columns if col != 'context_length']
        for col in columns_to_rename:
            data = data.rename(columns={col: f"{col}_{dataset_name}"})
        
        # Create plots
        create_comparison_plots(data, dataset_names, main_plot_file, silent=silent)
        #create_detailed_metrics_plots(data, dataset_names, detailed_plot_file, silent=silent)
        
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
        # Detailed analysis
        #analyze_performance_ranges(data, dataset_names)
        
        # Summary insights
        #create_summary_analysis(data, dataset_names)
        
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