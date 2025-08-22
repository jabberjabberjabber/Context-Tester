# Context Window Testing

This project utilizes a novel method for evaluated an LLMs ability to utilize different different sized context windows. Instead of checking for recall or correctness, it fills the context with a text and asks the model to continue the text as if it were the original writer. It then evaluates the model's output using basic metrics for readability, including sentence length, estimated grade level, and word variability and complexity.

The purpose is not to provide a benchmark that is definitive, but to provide data points for comparison so that one can see how changes to the model weights and processing effects its creative output across different corpus sizes.

A distinct feature of this test method is the use of a deterministic and static start point for each text and model. Instead of floating through a text going forward by adding tokens to the context window and progressing linearly through the story, the text is pre-tokenized and the largest point in the context window is used as the end of the prompt as sent to the model. The model starts at this spot every time to generate new text, and as tokens are added to increase the context, the earlier parts of the text is used is a reverse linear fashion.

In other words, if we start the test by continuing the story at the beginning of chapter 5, instead of adding tokens and then starting at chapter 6, then chapter 7, and so on, we add chapter 4, then chapter 3, until we reach the beginning. This way we always test from the same origin point. 

## Overview

This repository contains two primary analysis tools:

1. **Readability Degradation Tester** (`main.py`) - Measures how LLM output quality degrades as context length increases
2. **Performance Comparison Tool** (`generate_plot.py`) - Plots the data on to two graphs

## Installation

### Prerequisites

- A large text to use as the basis for continuation
- Python 3.8 or higher
- A running LLM API server (compatible with OpenAI chat completions format)

### Dependencies

CLone repo

```
git clone https://github.com/jabberjabberjabber/context-tester
cd context-tester
```

Install UV and sync:

```bash
pip install uv
uv sync
```

Ensure you have a running inference instance running an OpenAI endpoint that you can connect to. Ensure that you have a properly formatted creative text such as a novel which has at least enough text in it to max out the tokens you are testing for.  

Run data collection:

```
uv run main.py crime_english.txt --api-url http://localhost:5001
```

Wait for the tests to complete. You should now have a a csv file and two png files in the directory with the name of the model and the text in it containing the data plots.

If you want to compare data for more than one model, run the test for each model and then use generate_plots to run the comparison using the csv files:

```
uv run generate_plots.py name-of-first-csv-file.csv name-of-second-csv-file.csv 
```
You can put as many as you like and it will plot the data for each of them onto the same graphs.

## Input Text

Texts can be any type supported by extractous such as txt or pdf or html. It can be any formatting but better results are obtained if the paragraphs are separated by 2 blank lines and there is now introduction, index, or any other text in it except for the story and chapter headings.

## Detailed Usage

### Main Tester
```bash
usage: main.py [-h] [--api-url API_URL] [--api-password API_PASSWORD] [--word-list WORD_LIST] [--max-context MAX_CONTEXT] [--rounds ROUNDS] [--divisions DIVISIONS] [--model-name MODEL_NAME]
               [--max-tokens MAX_TOKENS] [--temp TEMP] [--top-k TOP_K] [--top-p TOP_P] [--min-p MIN_P] [--rep-pen REP_PEN] [--start-context START_CONTEXT]
               input_file

Test LLM readability degradation across context lengths with fixed continuation point

positional arguments:
  input_file            Path to reference text file (any format supported by extractous)

options:
  -h, --help            show this help message and exit
  --api-url API_URL     API URL for the LLM service
  --api-password API_PASSWORD
                        API key/password if required
  --word-list WORD_LIST
                        Path to Dale-Chall easy words list
  --max-context MAX_CONTEXT
                        Maximum context length to test (auto-detect if not specified)
  --rounds ROUNDS       Number of test rounds per context length (default: 3)
  --divisions DIVISIONS
                        Number of context divisions between tiers as a power of 2
  --model-name MODEL_NAME
                        Override model name (auto-detected if not provided)
  --max-tokens MAX_TOKENS
                        Maximum tokens to generate (default: 512)
  --temp TEMP           Generation temperature (default: 1.0)
  --top-k TOP_K         Top-k sampling (default: 100)
  --top-p TOP_P         Top-p sampling (default: 1.0)
  --min-p MIN_P         Min-p sampling (default: 0.1)
  --rep-pen REP_PEN     Repetition penalty (default: 1.01)
  --start-context START_CONTEXT
                        Starting context size in tokens (skip smaller sizes)

Examples:
  python main.py novel.txt --api-url http://localhost:5001
  python main.py document.pdf --max-context 16384 --model-name "MyModel"
  python main.py text.txt --word-list dale_chall_words.txt --rounds 3
  python main.py novel.txt --rounds 5 --divisions 2 --temp 0.8 --top-k 50
  python main.py text.txt --max-tokens 1024 --rep-pen 1.05 --min-p 0.05
  python main.py novel.txt --start-context 8192  # Skip testing small contexts
```

Notes:

**Divisions** allow you to add more datapoints to the normal span of context windows by adding more continuations in between. For example you normally have [1024, 2048], etc as data points; setting divisions to be 1 would give you [1024, a, 2048] where 'a' is an equidistant number of tokens between 1024 and 2048. These tokens will always be a power of two and divisions must also be a power of 2.
    
**Rounds** are the number of times a test is repeated at each tier. They are averaged out to mitigate the randomness of LLM generations. At least 3 are recommended.
 
## Understanding the Metrics

## Reading the Graphs

This is pretty simple. They should be flat. Any move up or down means the model is being inconsistent. But here is a breakdown:

**Left hand** graph goes up with the model outputs more diverse sentences and vocabulary. This indicates it is being more creative. It could also indicate it is generating well structured varied gibberish.

**Right hand** graph goes up when the model outputs more simple words with more predictable text. This indicates that it is degrading by choosing to use words that are less descriptive and more generic.

So: **Left hand** indicates *creativity* and **right hand** indicates *degredation*.

The second page of graphs should be self evident.
 
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
