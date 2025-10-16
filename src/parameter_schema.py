#!/usr/bin/env python3
"""
Single source of truth for all benchmark parameters.

This schema is used to automatically generate:
- CLI argument parsing (config.py)
- GUI fields (benchmark_gui.py)
- Command building
- Settings persistence
"""

PARAMETER_SCHEMA = {
    # Input/Output
    'input_file': {
        'type': 'positional',
        'help': 'Path to reference text OR results directory for analysis/reanalysis',
        'nargs': '?',
        'gui': False  # Handled separately with file browser
    },

    # API Configuration
    'api_url': {
        'type': 'str',
        'default': 'http://localhost:5001',
        'help': 'API base URL (e.g., http://localhost:5001)',
        'gui_label': 'API URL',
        'gui_section': 'api',
        'gui_width': None,  # Full width
        'gui_hint': None
    },
    'api_password': {
        'type': 'str',
        'default': None,
        'help': 'API key/bearer token',
        'gui_label': 'API Key',
        'gui_section': 'api',
        'gui_show': '*',  # Password field
        'gui_width': None,
        'gui_hint': None
    },
    'model_name': {
        'type': 'str',
        'default': None,
        'help': 'Model name',
        'gui_label': 'Model Name',
        'gui_section': 'api',
        'gui_type': 'combobox',  # Dropdown
        'gui_width': None,
        'gui_hint': None
    },
    'tokenizer_model': {
        'type': 'str',
        'default': None,
        'help': 'HuggingFace tokenizer model name',
        'gui_label': 'Tokenizer Model',
        'gui_section': 'api',
        'gui_type': 'combobox',
        'gui_width': None,
        'gui_hint': None
    },
    'embedding_model': {
        'type': 'str',
        'default': None,
        'help': 'Embedding model for NVIDIA NIM',
        'gui': False  # Hidden in GUI
    },

    # Test Parameters
    'max_context': {
        'type': 'int',
        'default': 131072,
        'help': 'Maximum context length',
        'gui_label': 'Max Context',
        'gui_section': 'test',
        'gui_type': 'combobox',
        'gui_values': ['2048', '4096', '8192', '16384', '32768', '65536', '131072', '262144'],
        'gui_width': 15,
        'gui_hint': 'tokens'
    },
    'start_context': {
        'type': 'int',
        'default': 2048,
        'help': 'Starting context size',
        'gui_label': 'Start Context',
        'gui_section': 'test',
        'gui_width': 15,
        'gui_hint': 'tokens (optional)'
    },
    'rounds': {
        'type': 'int',
        'default': 3,
        'help': 'Number of rounds per context size',
        'gui_label': 'Rounds',
        'gui_section': 'test',
        'gui_width': 15,
        'gui_hint': 'generations per context size'
    },
    'divisions': {
        'type': 'int',
        'default': 1,
        'help': 'Subdivisions between power-of-2 sizes',
        'gui_label': 'Divisions',
        'gui_section': 'test',
        'gui_type': 'combobox',
        'gui_values': ['1', '2', '4', '8', '16', '32', '64', '128', '256'],
        'gui_width': 15,
        'gui_hint': 'test point divisions'
    },
    'max_retries': {
        'type': 'int',
        'default': 2,
        'help': 'Maximum retry attempts on failure',
        'cli_name': 'max-retries',
        'gui_label': 'Max Retries',
        'gui_section': 'test',
        'gui_width': 15,
        'gui_hint': 'times to retry if there is a failure'
    },
    'model_id': {
        'type': 'str',
        'default': None,
        'help': 'Model identifier for organizing results',
        'cli_name': 'model-id',
        'gui_label': 'Model ID',
        'gui_section': 'test',
        'gui_width': None,
        'gui_hint': None
    },

    # Generation Parameters
    'max_tokens': {
        'type': 'int',
        'default': 1024,
        'help': 'Max tokens to generate',
        'cli_name': 'max-tokens',
        'gui_label': 'Max Tokens',
        'gui_section': 'generation',
        'gui_width': 15,
        'gui_hint': 'max output tokens'
    },
    'temperature': {
        'type': 'float',
        'default': 0.0,
        'help': 'Sampling temperature',
        'cli_name': 'temp',  # For backward compatibility with existing CLI
        'gui_label': 'Temperature',
        'gui_section': 'generation',
        'gui_width': 15,
        'gui_hint': None
    },
    'top_k': {
        'type': 'int',
        'default': 1,
        'help': 'Top-k sampling',
        'cli_name': 'top-k',
        'gui_label': 'Top K',
        'gui_section': 'generation',
        'gui_width': 15,
        'gui_hint': '(where supported)'
    },
    'top_p': {
        'type': 'float',
        'default': 0.01,
        'help': 'Top-p (nucleus) sampling',
        'cli_name': 'top-p',
        'gui_label': 'Top P',
        'gui_section': 'generation',
        'gui_width': 15,
        'gui_hint': None
    },
    'rep_pen': {
        'type': 'float',
        'default': None,
        'help': 'Repetition penalty',
        'cli_name': 'rep-pen',
        'gui_label': 'Rep Penalty',
        'gui_section': 'generation',
        'gui_width': 15,
        'gui_hint': '(where supported)'
    },
    'min_p': {
        'type': 'float',
        'default': None,
        'help': 'Minimum probability threshold (min-p sampling)',
        'cli_name': 'min-p',
        'gui_label': 'Min P',
        'gui_section': 'generation',
        'gui_width': 15,
        'gui_hint': '(where supported)'
    },
    'no_think': {
        'type': 'bool',
        'default': True,
        'help': 'Disable thinking for reasoning models',
        'cli_name': 'no-think',
        'gui_label': 'No Think',
        'gui_section': 'generation',
        'gui_type': 'checkbox',
        'gui_width': None,
        'gui_hint': '(where supported)'
    },
    'ignore_min_tokens': {
        'type': 'bool',
        'default': False,
        'help': 'Ignore minimum token requirements',
        'cli_name': 'ignore-min-tokens',
        'gui_label': 'No Min Tokens',
        'gui_section': 'generation',
        'gui_type': 'checkbox',
        'gui_width': None,
        'gui_hint': 'do not count output tokens'
    },
    'seed': {
        'type': 'int',
        'default': 0,
        'help': 'Seed for deterministic generation (where supported)',
        'gui_label': 'Seed',
        'gui_section': 'generation',
        'gui_width': 15,
        'gui_hint': '(where supported)'
    },

    # Analysis Modes
    'analyze': {
        'type': 'bool',
        'default': False,
        'help': 'Run analysis on existing results',
        'gui': False  # Not in GUI
    },
    'reanalyze': {
        'type': 'bool',
        'default': False,
        'help': 'Reanalyze with new embeddings',
        'gui': False  # Not in GUI
    },

    # Misc
    'word_list': {
        'type': 'str',
        'default': 'easy_words.txt',
        'help': 'Path to Dale-Chall word list',
        'cli_name': 'word-list',
        'gui': False  # Not in GUI
    },
}


def get_cli_name(param_name, schema_entry):
    """Get CLI argument name (with dashes)."""
    return schema_entry.get('cli_name', param_name.replace('_', '-'))


def get_var_name(param_name):
    """Get Python variable name (with underscores)."""
    return param_name


def get_gui_params():
    """Get parameters that should appear in GUI."""
    return {k: v for k, v in PARAMETER_SCHEMA.items()
            if v.get('gui', True) and v.get('type') != 'positional'}


def get_gui_params_by_section(section):
    """Get GUI parameters for a specific section."""
    return {k: v for k, v in get_gui_params().items()
            if v.get('gui_section') == section}
