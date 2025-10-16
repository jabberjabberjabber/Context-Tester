#!/usr/bin/env python3
"""
Configuration and argument parsing for Context Tester.

Handles command-line arguments, validation, and configuration management.
"""

import argparse
from pathlib import Path
from typing import Optional, Dict, Any

from .parameter_schema import PARAMETER_SCHEMA, get_cli_name


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


def create_argument_parser():
    """Create argument parser from parameter schema (auto-generated)."""
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

    # Dynamically add arguments from schema
    for param_name, spec in PARAMETER_SCHEMA.items():
        cli_name = get_cli_name(param_name, spec)
        param_type = spec.get('type')
        default = spec.get('default')
        help_text = spec.get('help')

        if param_type == 'positional':
            # Positional argument
            parser.add_argument(
                param_name,
                nargs=spec.get('nargs', None),
                help=help_text
            )
        elif param_type == 'bool':
            # Boolean flag
            parser.add_argument(
                f'--{cli_name}',
                action='store_true',
                default=default,
                help=help_text
            )
        elif param_type == 'int':
            # Integer argument
            parser.add_argument(
                f'--{cli_name}',
                type=int,
                default=default,
                help=help_text
            )
        elif param_type == 'float':
            # Float argument
            parser.add_argument(
                f'--{cli_name}',
                type=float,
                default=default,
                help=help_text
            )
        else:  # str
            # String argument
            parser.add_argument(
                f'--{cli_name}',
                default=default,
                help=help_text
            )

    return parser


def parse_arguments():
    """Parse and validate command-line arguments (auto-generated from parameter_schema)."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Get API password from environment if not provided via argument
    if not args.api_password:
        args.api_password = get_api_key_from_env()

    validate_arguments(args)

    return args


def parse_args(argv_list):
    """Parse arguments from a list (for GUI/programmatic use - auto-generated from parameter_schema).

    Args:
        argv_list: List of argument strings (like sys.argv[1:])

    Returns:
        Parsed arguments namespace
    """
    parser = create_argument_parser()
    args = parser.parse_args(argv_list)

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
