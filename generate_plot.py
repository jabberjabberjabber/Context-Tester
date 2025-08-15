#!/usr/bin/env python3
"""
Performance Comparison Tool

Analyzes and visualizes performance differences between ROPE and full context configurations
(rope and full_context) across different context lengths. Saves plots as PNG files.

Usage:
    python analysis.py <rope_file.csv> <full_context_file.csv>
    python analysis.py data/rope.csv data/full_context.csv --output results.png
    python analysis.py data/rope.csv data/full_context.csv --no-plots

Requirements:
    pip install pandas matplotlib numpy
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from matplotlib.patches import Rectangle

def load_and_compare_data(rope_file, full_context_file):
    """ Load and merge the rope and full_context datasets for comparison.
    
    Args:
        rope_file: Path to rope CSV file
        full_context_file: Path to full_context CSV file
    
    Returns:
        Merged dataframe with both rope and full_context metrics
    """
    # Load the data
    rope_data = pd.read_csv(rope_file)
    full_context_data = pd.read_csv(full_context_file)
    
    # Filter out null context lengths and merge on context_length
    rope_clean = rope_data[rope_data['context_length'].notna()].copy()
    full_context_clean = full_context_data[full_context_data['context_length'].notna()].copy()
    
    # Merge datasets
    merged = pd.merge(rope_clean, full_context_clean, on='context_length', suffixes=('_rope', '_full_context'))
    
    return merged.sort_values('context_length')

def create_comparison_plots(data, output_file='comparison.png', dpi=300):
    """ Create comprehensive comparison plots for rope vs full_context.
    
    Args:
        data: Merged dataframe with rope and full_context metrics
        output_file: Path for output PNG file
        dpi: Resolution for output image
    """
    # Set backend to Agg for headless operation
    import matplotlib
    matplotlib.use('Agg')
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('ROPE vs Full Conext Performance Comparison', fontsize=16, fontweight='bold')
    
    metrics = [
        ('cloze_score', 'Cloze Score', 'Score'),
        ('pct_unfamiliar_words', 'Unfamiliar Words', 'Percentage'),
        ('vocabulary_diversity', 'Vocabulary Diversity', 'Diversity Score'),
        ('continuation_length', 'Continuation Length', 'Tokens'),
        ('avg_sentence_length', 'Average Sentence Length', 'Words'),
        ('sentence_length_variance', 'Sentence Length Variance', 'Variance')
    ]
    
    for i, (metric, title, ylabel) in enumerate(metrics):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # Convert percentage if needed
        rope_values = data[f'{metric}_rope'] * (100 if metric == 'pct_unfamiliar_words' else 1)
        full_context_values = data[f'{metric}_full_context'] * (100 if metric == 'pct_unfamiliar_words' else 1)
        
        # Plot lines
        ax.plot(data['context_length'], rope_values, 'o-', color='#2563eb', 
                linewidth=3, markersize=6, label='rope')
        ax.plot(data['context_length'], full_context_values, 's-', color='#dc2626', 
                linewidth=3, markersize=6, label='full_context')
        
        # Formatting
        ax.set_xscale('log')
        ax.set_xlabel('Context Length')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Format x-axis labels
        ax.set_xticks(data['context_length'])
        ax.set_xticklabels([f'{int(x/1000)}K' if x >= 1000 else str(int(x)) 
                           for x in data['context_length']])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    plt.close()

def analyze_performance_ranges(data):
    """ Analyze which model performs better at different context ranges.
    
    Args:
        data: Merged dataframe with rope and full_context metrics
    """
    print("=" * 80)
    print("PERFORMANCE ANALYSIS BY CONTEXT LENGTH")
    print("=" * 80)
    
    for _, row in data.iterrows():
        ctx = int(row['context_length'])
        print(f"\nContext Length: {ctx:,}")
        
        # Cloze score comparison
        cloze_diff = row['cloze_score_full_context'] - row['cloze_score_rope']
        cloze_winner = "full_context" if cloze_diff > 0 else "rope"
        print(f"  Cloze Score    - rope: {row['cloze_score_rope']:.2f}, full_context: {row['cloze_score_full_context']:.2f} "
              f"(Winner: {cloze_winner}, Δ: {cloze_diff:+.2f})")
        
        # Unfamiliar words (lower is better)
        unfam_diff = row['pct_unfamiliar_words_full_context'] - row['pct_unfamiliar_words_rope']
        unfam_winner = "rope" if unfam_diff > 0 else "full_context"
        print(f"  Unfamiliar %   - rope: {row['pct_unfamiliar_words_rope']*100:.1f}%, "
              f"full_context: {row['pct_unfamiliar_words_full_context']*100:.1f}% "
              f"(Winner: {unfam_winner}, Δ: {unfam_diff*100:+.1f}%)")
        
        # Vocabulary diversity
        vocab_diff = row['vocabulary_diversity_full_context'] - row['vocabulary_diversity_rope']
        vocab_winner = "full_context" if vocab_diff > 0 else "rope"
        print(f"  Vocab Diversity- rope: {row['vocabulary_diversity_rope']:.3f}, "
              f"full_context: {row['vocabulary_diversity_full_context']:.3f} "
              f"(Winner: {vocab_winner}, Δ: {vocab_diff:+.3f})")
        
        # Continuation length
        cont_diff = row['continuation_length_full_context'] - row['continuation_length_rope']
        print(f"  Continuation   - rope: {row['continuation_length_rope']}, "
              f"full_context: {row['continuation_length_full_context']} tokens (Δ: {cont_diff:+})")

def create_summary_analysis(data):
    """ Create summary statistics and insights.
    
    Args:
        data: Merged dataframe with rope and full_context metrics
    """
    print("\n" + "=" * 80)
    print("SUMMARY INSIGHTS")
    print("=" * 80)
    
    # Overall averages
    rope_avg_cloze = data['cloze_score_rope'].mean()
    full_context_avg_cloze = data['cloze_score_full_context'].mean()
    rope_avg_cont = data['continuation_length_rope'].mean()
    full_context_avg_cont = data['continuation_length_full_context'].mean()
    
    print(f"Overall Performance:")
    print(f"  Average Cloze Score - rope: {rope_avg_cloze:.2f}, full_context: {full_context_avg_cloze:.2f}")
    print(f"  Average Continuation Length - rope: {rope_avg_cont:.0f}, full_context: {full_context_avg_cont:.0f} tokens")
    print(f"  full_context generates {((full_context_avg_cont - rope_avg_cont) / rope_avg_cont * 100):+.1f}% longer text")
    
    # Context-specific advantages
    print(f"\nContext-Specific Advantages:")
    
    # Find where each model wins on cloze score
    rope_cloze_wins = data[data['cloze_score_rope'] > data['cloze_score_full_context']]['context_length'].tolist()
    full_context_cloze_wins = data[data['cloze_score_full_context'] > data['cloze_score_rope']]['context_length'].tolist()
    
    print(f"  rope better cloze scores at: {[f'{int(x/1000)}K' if x >= 1000 else str(int(x)) for x in rope_cloze_wins]}")
    print(f"  full_context better cloze scores at: {[f'{int(x/1000)}K' if x >= 1000 else str(int(x)) for x in full_context_cloze_wins]}")
    
    # Peak performance
    rope_best_cloze_ctx = data.loc[data['cloze_score_rope'].idxmax(), 'context_length']
    full_context_best_cloze_ctx = data.loc[data['cloze_score_full_context'].idxmax(), 'context_length']
    
    print(f"  rope peak cloze performance at: {int(rope_best_cloze_ctx):,} context")
    print(f"  full_context peak cloze performance at: {int(full_context_best_cloze_ctx):,} context")

def parse_arguments():
    """ Parse command line arguments for CSV file inputs.
    
    Returns:
        Parsed arguments containing file paths
    """
    parser = argparse.ArgumentParser(
        description='Compare ROPE and FULL CONTEXT performance across different context lengths'
    )
    
    parser.add_argument('rope_file', 
                       help='Path to the ROPE CSV file')
    parser.add_argument('full_context_file', 
                       help='Path to the FULL CONTEXT CSV file')
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
    
    try:
        # Load and merge data
        print(f"Loading data from {args.rope_file} and {args.full_context_file}...")
        data = load_and_compare_data(args.rope_file, args.full_context_file)
        print(f"Loaded {len(data)} context length comparisons")
        
        # Create visualizations (unless disabled)
        if not args.no_plots:
            print("Creating comparison plots...")
            create_comparison_plots(data, args.output, args.dpi)
        else:
            print("Skipping plots (--no-plots specified)")
        
        # Detailed analysis
        analyze_performance_ranges(data)
        
        # Summary insights
        create_summary_analysis(data)
        
    except FileNotFoundError as e:
        print(f"Error: Could not find CSV files.")
        print(f"Make sure '{args.rope_file}' and '{args.full_context_file}' exist and are accessible.")
        print(f"Details: {e}")
    except Exception as e:
        print(f"Error processing data: {e}")

if __name__ == "__main__":
    main()