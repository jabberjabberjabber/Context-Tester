# Context Window Testing

This project utilizes a novel method for evaluated an LLMs ability to utilize different different sized context windows. Instead of checking for recall or correctness, it fills the context with a text and asks the model to continue the text as if it were the original writer. It then evaluates the model's output using basic metrics for readability, including sentence length, estimated grade level, and word variability and complexity.

The purpose is not to provide a benchmark that is definitive, but to provide data points for comparison so that one can see how changes to the model weights and processing effects its creative output across different corpus sizes.

A distinct feature of this test method is the use of a deterministic and static start point for each text and model. Instead of floating through a text going forward by adding tokens to the context window and progressing linearly through the story, the text is pre-tokenized and the largest point in the context window is used as the end of the prompt as sent to the model. The model starts at this spot every time to generate new text, and as tokens are added to increase the context, the earlier parts of the text is used is a reverse linear fashion.

In other words, if we start the test by continuing the story at the beginning of chapter 5, instead of adding tokens and then starting at chapter 6, then chapter 7, and so on, we add chapter 4, then chapter 3, until we reach the beginning. This way we always test from the same origin point. 

## Overview

This repository contains two primary analysis tools:

1. **Readability Degradation Tester** (`main.py`) - Measures how LLM output quality degrades as context length increases
2. **Performance Comparison Tool** (`generate_plot.py`) - Compares ROPE vs full context configurations across multiple metrics

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Tools Overview](#tools-overview)
- [Detailed Usage](#detailed-usage)
- [Understanding the Metrics](#understanding-the-metrics)
- [Output Formats](#output-formats)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- A large text to use as the basis for continuation
- Python 3.8 or higher
- A running LLM API server (compatible with OpenAI chat completions format)

### Dependencies

Install required packages:

```bash
pip install pandas matplotlib numpy requests beautifulsoup4 extractous
```

## Quick Start

### Testing Readability Degradation

```bash
# Basic test with a novel or long text
python main.py novel.txt --api-url http://localhost:5001

# Multi-round testing for statistical reliability
python main.py document.pdf --rounds 5 --output degradation_results.csv

# Custom context limits
python main.py text.txt --max-context 16384 --rounds 3 -divisions 2
```

### Comparing ROPE vs Full Context

```bash
# Generate comparison plots
python generate_plot.py rope_results.csv full_context_results.csv

# Custom output and analysis only
python generate_plot.py rope_data.csv full_context_data.csv --output comparison.png --no-plots
```

## Tools Overview

### Readability Degradation Tester (`main.py`)

**Purpose**: Identifies at what context lengths LLM output quality begins to degrade by measuring readability complexity.

**Key Features**:
- Tests multiple context lengths (powers of 2: 1K, 2K, 4K, 8K, 16K, 32K+)
- Intermediary tiers with divisions
- Multi-round testing with statistical averaging
- Support for any text format via extractous

### Performance Comparison Tool (`generate_plot.py`)

**Key Features**:
- Comprehensive metric comparison across context lengths
- Statistical analysis and winner identification
- High-quality visualization generation
- Performance range analysis
- Summary insights and recommendations

## Detailed Usage

```bash
python main.py [text] [options]
```

**Arguments**:
- `text`: Path to a long creative text that the model can continue any part of (supports PDF, DOCX, TXT, etc.)

**Options**:
- `--api-url`: Koboldcpp API endpoint (default: `http://localhost:5001`)
- `--api-password`: API authentication key
- `--word-list`: Path to Dale-Chall word list (default: `easy_words.txt`)
- `--max-context`: Maximum context length to test (auto-detected)
- `--rounds`: Number of test rounds per context length (default: 3)
- `--divisions`: Split the power of two context tiers into this many parts to create more test datapoints
- `--output`: CSV output file path (required if plotting a graph)

**Multi-Round Testing**:
Running multiple rounds per context length provides statistical reliability:
- Reduces impact of generation randomness
- Provides standard deviation metrics
- Enables confidence interval analysis
- Recommended: 3-15 rounds for production analysis

### Performance Comparison Tool

```bash
python generate_plot.py [csv-file] [csv-file] [...] [options]
```

**Arguments**:
- `csv_file`: Any a CSV file output by the main script containing single datapoints per context window (not the detailed file) 

**Options**:
- `--output`: Output PNG filename (default: `comparison.png`)
- `--no-plots`: Skip plot generation (analysis only)
- `--dpi`: Output image resolution (default: 300)

## Understanding the Metrics

### Readability Metrics

**Cloze Score**: Primary readability indicator
- *LOWER IS BETTER*
- Range: 10-64 (higher = more readable)
- Formula: `64 - (95 × pct_unfamiliar_words) - (0.69 × avg_sentence_length)`
- Based on Dale-Chall readability research

**Vocabulary Diversity**: `unique_words / total_words`
- Range: 0.0-1.0 (higher = more diverse)
- Measures repetitiveness and word choice variety

**Sentence Length Variance**: Statistical variance of sentence lengths
- Higher values indicate more varied sentence structure
- Lower values suggest repetitive patterns

**Unfamiliar Words Percentage**: Words not in Dale-Chall easy list
- Lower percentages indicate simpler vocabulary
- Sudden increases may signal degradation

### Performance Comparison Metrics

The comparison tool analyzes six key metrics:
- **Cloze Score**: Text readability quality
- **Unfamiliar Words**: Vocabulary complexity
- **Vocabulary Diversity**: Word choice variety  
- **Continuation Length**: Generated text length
- **Average Sentence Length**: Sentence structure consistency
- **Sentence Length Variance**: Structural diversity
