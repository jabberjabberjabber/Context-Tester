#!/usr/bin/env python3
"""
Configuration and argument parsing for Context Tester.

Handles command-line arguments, validation, and configuration management.
"""

import argparse
from pathlib import Path
from typing import Optional, Dict, Any


def get_api_key_from_env() -> Optional[str]:
    """Get API key from environment variables."""
    import os
    # Check common environment variable names
    return (
        os.environ.get('API_KEY') or
        os.environ.get('API_PASSWORD') or
        os.environ.get('OPENAI_API_KEY') or
        os.environ.get('NVIDIA_API_KEY') or
        os.environ.get('NVAPI_KEY')
    )


def parse_arguments():
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Test LLM readability degradation across context lengths",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py novel.txt --api-url http://localhost:5001
  python main.py document.pdf --max-context 16384 --model-name "MyModel"
  python main.py text.txt --rounds 3 --start-context 8192
  python main.py --analyze results/model-text-20241201-123456
  python main.py --reanalyze results/model-text-20241201-123456
Important:  
  Always set top-k, top-p, rep-pen when using KoboldCpp.
  If you set a top-k, top-p, or rep-pen when using an API that does not support it, it will fail.
"""
    )

    # Input file or results directory
    parser.add_argument(
        'input_file',
        nargs='?',
        help='Path to reference text OR results directory for analysis/reanalysis'
    )

    # Mode selection
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Run analysis on existing results directory'
    )

    parser.add_argument(
        '--reanalyze',
        action='store_true',
        help='Re-run text analysis on stored continuation texts'
    )

    # API configuration
    parser.add_argument(
        '--api-url',
        default='http://localhost:5001',
        help='API URL for the LLM service (default: http://localhost:5001)'
    )

    parser.add_argument(
        '--api-password',
        default=None,
        help='API key/password if required (or set API_KEY/API_PASSWORD env variable)'
    )

    # Model configuration
    parser.add_argument(
        '--model-name',
        default=None,
        help='Override model name (auto-detected for KoboldCpp)'
    )

    parser.add_argument(
        '--tokenizer-model',
        default=None,
        help='HuggingFace tokenizer model name (e.g., "meta-llama/Llama-3.1-8B-Instruct")'
    )

    parser.add_argument(
        '--embedding-model',
        default="nvidia/nv-embed-v1",
        help='Embedding model name (default: nvidia/nv-embed-v1)'
    )

    parser.add_argument(
        '--max-context',
        type=int,
        default=None,
        help='Maximum context length to test (required when using --tokenizer-model)'
    )

    # Test configuration
    parser.add_argument(
        '--word-list',
        default='easy_words.txt',
        help='Path to Dale-Chall easy words list (default: easy_words.txt)'
    )

    parser.add_argument(
        '--rounds',
        type=int,
        default=10,
        help='Number of test rounds per tier (default: 10, minimum 3 recommended)'
    )

    parser.add_argument(
        '--divisions',
        type=int,
        default=1,
        help='Divide tiers (must be power of 2, default: 1)'
    )

    parser.add_argument(
        '--start-context',
        type=int,
        default=2048,
        help='Starting context size in tokens (default: 2048)'
    )

    parser.add_argument(
        '--max-retries',
        type=int,
        default=2,
        help='Maximum retries for outlier contexts (default: 2)'
    )

    parser.add_argument(
        '--ignore-min-tokens',
        action="store_true",
        help='Ignore minimum tokens required for successful generation'
    )
    parser.add_argument(
        '--no-think',
        action="store_true",
        help='Ask the model not to use thinking tags'
    )
    # Generation parameters
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=1024,
        help='Maximum tokens to generate (default: 1024)'
    )

    parser.add_argument(
        '--temp',
        type=float,
        default=0.1,
        help='Generation temperature (default: 0.1)'
    )

    parser.add_argument(
        '--top-k',
        type=int,
        default=None,
        help='Top-k sampling'
    )

    parser.add_argument(
        '--top-p',
        type=float,
        default=None,
        help='Top-p sampling'
    )
    parser.add_argument(
        '--rep-pen',
        type=float,
        default=None,
        help='Rep-pen sampling'
    )

    # Output configuration
    parser.add_argument(
        '--model-id',
        default=None,
        help='Optional model identifier for result filenames (e.g., "v2", "fine-tuned")'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Seed for deterministic generation (where supported)'
    )

    args = parser.parse_args()

    # Get API password from environment if not provided via argument
    if not args.api_password:
        args.api_password = get_api_key_from_env()

    validate_arguments(args)

    return args


def validate_arguments(args):
    """Validate argument combinations and constraints."""
    from src.readability_tests import is_power_of_two

    # Validate mode combinations
    if args.analyze and args.reanalyze:
        raise ValueError("Cannot specify both --analyze and --reanalyze")

    if args.analyze or args.reanalyze:
        if not args.input_file or not Path(args.input_file).exists():
            raise ValueError(f"Must specify valid results directory: {args.input_file}")
        return

    # Validate data collection mode
    if not args.input_file:
        raise ValueError("Must specify input file for data collection")

    if not Path(args.input_file).exists():
        raise ValueError(f"Input file does not exist: {args.input_file}")

    if not is_power_of_two(args.divisions):
        raise ValueError(f"Divisions must be 1 or a power of 2 (2, 4, 8, etc.), got {args.divisions}")

    if args.start_context < int(args.max_tokens * 2):
        raise ValueError(
            f"Start context ({args.start_context}) must be at least 2x larger "
            f"than max tokens ({args.max_tokens})"
        )

    if args.tokenizer_model and not args.max_context:
        raise ValueError("Must specify --max-context when using --tokenizer-model")
    

def create_generation_params(args) -> Dict[str, Any]:
    """Create generation parameters dictionary from arguments."""
    if args.no_think:
        no_think = True
    else:
        no_think = False
        
    return {
        'max_tokens': args.max_tokens,
        'temperature': args.temp,
        'top_k': args.top_k,
        'top_p': args.top_p,
        'rep_pen': args.rep_pen,
        'no_think': no_think,
        'seed': args.seed
    }


def create_experiment_metadata(args, model_name: str, text_name: str) -> Dict[str, Any]:
    """Create experiment metadata dictionary."""
    from datetime import datetime

    return {
        'experiment_metadata': {
            'model_name': model_name,
            'model_id': args.model_id,
            'embedding_model': args.embedding_model,
            'tokenizer_model': args.tokenizer_model,
            'source_text_file': args.input_file,
            'source_text_name': text_name,
            'start_time': datetime.now().isoformat(),
            'num_rounds': args.rounds,
            'divisions': args.divisions,
            'start_context': args.start_context,
            'max_retries': args.max_retries,
            'generation_params': create_generation_params(args),
            'api_url': args.api_url
        }
    }
