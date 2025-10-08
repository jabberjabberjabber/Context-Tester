# Context Window Testing

This project utilizes a novel method for evaluating an LLM's ability to utilize different sized context windows. It is especially concerned with style, creativity and degradation of long form outputs as the amount of tokens fill larger context windows.

Instead of checking for recall or correctness, the test fills the context with a text and asks the model to continue the text as if it were the original writer. It then evaluates the model's output using basic metrics for readability, sentence length, and vocabulary diversity.

The purpose is not to provide a benchmark that is definitive, but to provide data points for comparison so that one can see how changes to the model weights and processing affects its creative output across different corpus sizes.

The script finds a natural break point in the text and then cuts out a set of tokens a bit larger than the largest test size. It will then expand the context window by adding new tokens going backwards from the continuation point. In this way the context window can expand with minimal effect on the model's creative choices. Any changes in style or creativity are then assumed to be from the larger number of tokens in the window, and not from stylistic choices due to the text changing. All of these operations are determined by using the actual tokens after converting the text using the model's own tokenizer, which allows for granular slicing of the context windows.

## Overview

This repository contains two primary analysis tools:

1. **Context Tester** (`main.py`) - Measures how LLM output quality degrades as context length increases
2. **Performance Comparison Tool** (`generate_plot.py`) - Creates comparison plots from test results

## Installation

### Prerequisites

- Python 3.13 or higher
- A large text to use as the basis for continuation (txt, pdf, or html)
- An OpenAI-compatible API endpoint (KoboldCpp, NVIDIA NIM, OpenAI, etc.)

### Setup

Clone the repository:

```bash
git clone https://github.com/jabberjabberjabber/Context-Tester
cd Context-Tester
```

Install UV and sync dependencies:

```bash
pip install uv
uv sync
```

### API Configuration

The tool supports multiple ways to provide API credentials:

**Option 1: Environment Variables (Recommended)**

```bash
# Set API key (checks in order: API_KEY, API_PASSWORD, OPENAI_API_KEY, NVIDIA_API_KEY, NVAPI_KEY)
export API_KEY=your-api-key-here

# For HuggingFace gated models (like Llama)
export HF_TOKEN=hf_your-token-here
```

**Option 2: Command Line Flag**

```bash
python main.py novel.txt --api-url https://api.endpoint.com --api-password your-api-key
```

## Quick Start

### Using KoboldCpp (Local)

Start KoboldCpp with your model:

```bash
koboldcpp --model modelname.gguf --contextsize 32768
```

Run the test:

```bash
uv run main.py middlemarch.txt --api-url http://localhost:5001
```

### Using NVIDIA NIM (Remote)

```bash
export API_KEY=nvapi-your-key-here
export HF_TOKEN=hf_your-token-here

uv run main.py middlemarch.txt \
  --api-url https://integrate.api.nvidia.com \
  --max-context 131072 \
  --tokenizer-model "microsoft/phi-4-mini-instruct" \
  --model-name "microsoft/phi-4-mini-instruct"
```

### Using OpenAI API

```bash
export OPENAI_API_KEY=sk-your-key-here

uv run main.py middlemarch.txt \
  --api-url https://api.openai.com \
  --model-name "gpt-4" \
  --max-context 128000
```

## API Compatibility

The tool works with any OpenAI-compatible API endpoint:

- **KoboldCpp**: Auto-detects model name and max context, uses API tokenization
- **NVIDIA NIM**: Requires `--tokenizer-model` and `--max-context` parameters
- **OpenAI**: Requires `--model-name` and `--max-context` parameters
- **Other OpenAI-compatible APIs**: Use `--api-url` with appropriate parameters

### Tokenizer Support

The system uses a unified tokenizer interface with automatic fallback:

1. **HuggingFace transformers** (primary) - Local tokenization with auto-discovery
2. **KoboldCpp API** (fallback) - Remote tokenization endpoint
3. **Tiktoken** (fallback) - OpenAI tokenization

For gated HuggingFace repositories (like Llama models), set the `HF_TOKEN` environment variable to automatically authenticate.

## Input Text

Texts can be any type supported by extractous such as txt, pdf, or html. It can be any formatting but better results are obtained if the paragraphs are separated by a blank line and there is no introduction, index, or any other text in it except the story and chapter headings.

## Running Tests

### Basic Test Execution

```bash
# Run test with default settings
uv run main.py novel.txt --api-url http://localhost:5001

# Custom context sizes and rounds
uv run main.py novel.txt --max-context 16384 --rounds 5 --divisions 1

# Generation parameters
uv run main.py text.txt --max-tokens 1024 --temp 0.8 --top-k 50

# Skip smaller context sizes
uv run main.py novel.txt --start-context 8192
```

### Test Parameters

- **`--rounds`**: Number of generations per context size (default: 10). Averaged to reduce randomness. Minimum 3 recommended, 10+ for statistical significance testing.
- **`--divisions`**: Add test points between power-of-2 context sizes (must be power of 2)
- **`--max-context`**: Maximum context length to test (auto-detected for KoboldCpp)
- **`--start-context`**: Skip testing context sizes below this value
- **`--model-id`**: Custom identifier for this test run (defaults to timestamp)

### Results Structure

Tests create a directory in `results/` with the format:

```
results/org-model-text-timestamp/
├── metadata.json                    # Experiment configuration
├── context_2048_results.json        # Results for each context size
├── context_4096_results.json
├── ...
├── degradation_analysis.json        # Statistical analysis
├── model-text-timestamp.csv         # Aggregate data for plotting
├── model-text-timestamp_generations.txt  # All LLM outputs
└── model-text-timestamp.png         # Performance graphs
```

## Analysis and Plotting

### Post-Test Analysis

After a test completes, results are automatically analyzed and plotted. You can reanalyze existing results:

```bash
# Reanalyze and regenerate plots
uv run main.py --analyze results/model-text-20250108-123456

# Reanalyze without regenerating plots
uv run main.py --reanalyze results/model-text-20250108-123456
```

### Comparison Plots

Compare multiple test runs:

```bash
# Compare using CSV files
uv run generate_plot.py model1-results.csv model2-results.csv model3-results.csv

# Compare using results directories
uv run generate_plot.py results/model1-test/ results/model2-test/

# Plot individual rounds alongside averages
uv run generate_plot.py results/model-test/ --plot-rounds

# Enhanced statistical analysis (composite scores, significance testing, effect sizes)
uv run generate_plot.py results/model-test/ --enhanced
```

**Plot Types:**

- **Standard Comparison**: Shows the four core metrics across context sizes for multiple models
- **Rounds Plotting** (`--plot-rounds`): Creates separate lines for each test round (R1, R2, R3, etc.) plus the average (AVG), visualizing variance between individual generations
- **Enhanced Statistical Analysis** (`--enhanced`): Creates a comprehensive 4-panel dashboard with:
  - Composite degradation score with 95% confidence intervals
  - Consistency/variance indicator (coefficient of variation)
  - Effect size visualization (rank-biserial correlation)
  - Statistical significance map with multiple comparison correction

## Understanding the Results

### The Four Metrics

The tool evaluates four key metrics:

1. **Vocabulary Diversity** (top-left) - Higher = less diverse wording (inverted y-axis)
2. **Cloze Score** (top-right) - Higher = more basic/predictable text
3. **Adjacent Sentence Similarity** (bottom-left) - Higher = more similar sentences
4. **Bigram Repetition Rate** (bottom-right) - Higher = more repetitive patterns

### Reading the Graphs

Ideally, all four graphs should show relatively flat lines across context sizes. Any significant movement up or down indicates inconsistency in the model's output as context grows.

**Good Result (Flat Lines):**

![Broken Tutu](Broken-Tutu-24B.Q6_K-middlemarch.txt-3r1d.png)

This shows consistent performance across all context sizes.

**Degradation Example:**

![Cohere Aya](Cohere_aya-23-8B-Q6_K-middlemarch.txt-3r1d.png)

The vocabulary diversity drops at 16K (becomes less diverse), while the Cloze score steadily rises (becomes more predictable). This indicates context-length-related degradation.

### Important Limitations

Since this test doesn't evaluate coherence, style accuracy, or instruction following, it should not be used as evidence of overall model capability. A good result could be achieved by the model generating well-structured, diverse nonsense.

The test is meant as a starting point and as a simple and easily readable indicator of output consistency across context sizes.

### Enhanced Statistical Analysis

The `--enhanced` mode provides rigorous statistical analysis for determining if degradation is real or just random variation:

**Panel 1: Composite Degradation Score**
- Combines all four metrics into a single 0-100 score (higher = more degradation)
- Shows 95% bootstrap confidence intervals
- Color-coded zones: green (no degradation), yellow (mild), red (severe)
- Significant degradation points marked with red X (sized by effect size)
- Displays Spearman correlation trend analysis

**Panel 2: Consistency Indicator**
- Coefficient of variation (CV) shows how consistent performance is at each context size
- Higher CV = less reliable/more variable performance
- Bars exceeding CV=0.15 threshold shown in red

**Panel 3: Effect Size Visualization**
- Shows magnitude of difference vs baseline using rank-biserial correlation
- Color-coded: green (small <0.3), yellow (medium 0.3-0.5), red (large >0.5)
- Effect size tells you if statistical significance is practically meaningful

**Panel 4: Statistical Significance Map**
- Mann-Whitney U test comparing each context size to baseline
- Includes Holm-Bonferroni multiple comparison correction
- Green circle = no change, yellow triangle = marginal (p≤0.05), red X = significant (p≤0.01)
- Summary shows: degradation detected (yes/no), max effect size, number of significant points

**When to use Enhanced mode:**
- You need statistical confidence that degradation is real, not random
- You're comparing models and need effect sizes, not just visual trends
- You have sufficient rounds (10+ recommended) for statistical power
- You want to publish/present results with proper statistical backing

## Text Choice

Different reference texts can produce different results. Here is a comparison of the same model tested with Crime and Punishment vs Middlemarch:

![Text choice](zai-org_GLM-4-9B-0414-Q6_K-crime_english_with-zai-org_GLM-4-9B-0414-Q6_K-3rounds_1divs-middlemarch.png)

## Detailed Usage

### main.py

```bash
usage: main.py [-h] [--api-url API_URL] [--api-password API_PASSWORD]
               [--tokenizer-model TOKENIZER_MODEL] [--word-list WORD_LIST]
               [--max-context MAX_CONTEXT] [--rounds ROUNDS]
               [--divisions DIVISIONS] [--model-name MODEL_NAME]
               [--model-id MODEL_ID] [--max-tokens MAX_TOKENS]
               [--temp TEMP] [--top-k TOP_K] [--top-p TOP_P]
               [--start-context START_CONTEXT] [--analyze ANALYZE]
               [--reanalyze REANALYZE]
               [input_file]

Test LLM readability degradation across context lengths with fixed continuation point

positional arguments:
  input_file            Path to reference text file (txt, pdf, html via extractous)

options:
  -h, --help            Show this help message and exit
  --api-url API_URL     API URL for OpenAI-compatible endpoint
  --api-password API_PASSWORD
                        API key/password (or use environment variables)
  --tokenizer-model TOKENIZER_MODEL
                        HuggingFace tokenizer model name (auto-detect if not specified)
  --word-list WORD_LIST
                        Path to Dale-Chall easy words list (default: easy_words.txt)
  --max-context MAX_CONTEXT
                        Maximum context length to test (auto-detect for KoboldCpp)
  --rounds ROUNDS       Number of test rounds per context length (default: 10)
  --divisions DIVISIONS
                        Number of context divisions between tiers as power of 2
  --model-name MODEL_NAME
                        Override model name (auto-detected for KoboldCpp)
  --model-id MODEL_ID   Custom test identifier (defaults to timestamp)
  --max-tokens MAX_TOKENS
                        Maximum tokens to generate (default: 1024)
  --temp TEMP           Generation temperature (default: 1.0)
  --top-k TOP_K         Top-k sampling (default: 100)
  --top-p TOP_P         Top-p sampling (default: 1.0)
  --start-context START_CONTEXT
                        Starting context size in tokens (skip smaller sizes)
  --analyze ANALYZE     Analyze existing results directory and regenerate plots
  --reanalyze REANALYZE
                        Reanalyze existing results without regenerating plots
```

### generate_plot.py

```bash
usage: generate_plot.py [-h] [--plot-rounds] [--enhanced] [--dpi DPI]
                        inputs [inputs ...]

Create comparison plots from context test results

positional arguments:
  inputs         CSV files or results directories to compare

options:
  -h, --help     Show this help message and exit
  --plot-rounds  Plot individual rounds alongside averages (requires results directories)
  --enhanced     Create enhanced plots with composite scores and statistical analysis
                 (requires single results directory, 10+ rounds recommended)
  --dpi DPI      Output image resolution (default: 300)
```

**Note:** The `--enhanced` flag requires exactly one results directory input and works best with 10+ rounds for statistical power.

## Examples

### Basic Testing

```bash
# Local KoboldCpp test
uv run main.py novel.txt --api-url http://localhost:5001

# NVIDIA NIM test with environment variables
export API_KEY=nvapi-your-key
export HF_TOKEN=hf_your-token
uv run main.py novel.txt \
  --api-url https://integrate.api.nvidia.com \
  --max-context 131072 \
  --tokenizer-model "microsoft/phi-4-mini-instruct"

# Custom test parameters
uv run main.py novel.txt \
  --rounds 5 \
  --divisions 1 \
  --temp 0.8 \
  --top-k 50 \
  --max-tokens 1024 \
  --start-context 8192

# Custom test ID (prevents accidental overwrites)
uv run main.py novel.txt \
  --api-url http://localhost:5001 \
  --model-id "fine-tune-v2"
```

### Analysis and Plotting

```bash
# Reanalyze existing results
uv run main.py --analyze results/model-text-20250108-123456

# Compare multiple models
uv run generate_plot.py \
  results/model1-test/model1-test.csv \
  results/model2-test/model2-test.csv \
  results/model3-test/model3-test.csv

# Compare using directories (cleaner)
uv run generate_plot.py results/model1-test/ results/model2-test/

# Plot individual rounds to see variance
uv run generate_plot.py results/model-test/ --plot-rounds

# Enhanced statistical analysis with composite scores
uv run generate_plot.py results/model-test/ --enhanced

# High-resolution enhanced analysis
uv run generate_plot.py results/model-test/ --enhanced --dpi 600
```

## Architecture

The codebase is modular with clear separation of concerns:

- **config.py** - Argument parsing and configuration management
- **tokenizer_utils.py** - Unified tokenization with automatic fallback
- **file_operations.py** - All file I/O and result management
- **benchmark_runner.py** - Test execution and retry logic
- **streaming_api.py** - API client wrapper
- **readability_tests.py** - Metrics calculation (Cloze, vocabulary diversity, etc.)
- **outlier_detection.py** - IQR-based outlier detection for retries
- **statistical_analysis.py** - Advanced statistical analysis (composite scores, significance testing, effect sizes)
- **main.py** - Core orchestration and text processing
- **generate_plot.py** - Plotting and visualization (standard and enhanced modes)

## Environment Variables

The following environment variables are supported:

**API Keys** (checked in order):
- `API_KEY`
- `API_PASSWORD`
- `OPENAI_API_KEY`
- `NVIDIA_API_KEY`
- `NVAPI_KEY`

**HuggingFace Token**:
- `HF_TOKEN` - For accessing gated models (Llama, etc.)

## Notes

**Divisions**: Allow you to add more data points to the normal span of context windows by adding more continuations in between. For example, you normally have [2048, 4096, 8192] as data points; setting divisions to 1 would give you [2048, 2896, 4096, 5793, 8192] where the middle values are equidistant powers of two. Divisions must be a power of 2.

**Rounds**: The number of times a test is repeated at each context size. They are averaged to mitigate the randomness of LLM generations. The default is 10 rounds, which provides sufficient statistical power for detecting moderate degradation effects. Minimum of 3 recommended for basic testing. Individual rounds can be plotted using `--plot-rounds` to visualize variance.

**Outlier Detection**: If a generation at a context size produces statistical outliers (based on IQR analysis), it will be automatically retried. This helps ensure consistent, reliable results.

**Lazy Directory Creation**: Output directories are only created after the first successful generation. This prevents empty folders from being created when API connection or configuration errors occur.

**Model ID**: Each test run has a unique identifier (timestamp by default, or custom via `--model-id`). If you try to run a test with an existing model ID, the tool will error to prevent accidental data loss. Either delete the old directory, use a different `--model-id`, or let the system auto-generate a timestamp.

