#!/usr/bin/env python3
"""
Metadata utilities for Context Tester.

Functions for formatting and displaying experiment metadata.
"""

from typing import Dict, Any, List


def format_metadata_for_tooltip(metadata: Dict[str, Any]) -> str:
    """Format metadata as compact tooltip text.

    Args:
        metadata: Full metadata dict

    Returns:
        Multi-line string suitable for tooltip display
    """
    exp_meta = metadata.get('experiment_metadata', {})
    gen_params = exp_meta.get('generation_params', {})

    lines = [
        f"  {key.replace('_', ' ').title()}: {value}" 
        for key, value in gen_params.items() 
        if value is not None
        ]
            

    lines += [
        f"  Model Name: {exp_meta.get('model_name', 'N/A')}",
        #f"  Model ID: {exp_meta.get('model_id', 'N/A')}",
        f"  Source Text: {exp_meta.get('source_text_name', 'N/A')}",
        f"  API: {exp_meta.get('api_url', 'N/A')}"
    ]

    return "\n".join(lines)


def format_metadata_for_popup(metadata: Dict[str, Any]) -> str:
    """Format metadata as detailed text for popup window.

    Args:
        metadata: Full metadata dict

    Returns:
        Multi-line formatted string with all metadata details
    """
    exp_meta = metadata.get('experiment_metadata', {})
    gen_params = exp_meta.get('generation_params', {})

    output = []

    # Experiment info
    output.append("=" * 60)
    output.append("EXPERIMENT INFORMATION")
    output.append("=" * 60)
    output.append(f"Model: {exp_meta.get('model_name', 'N/A')}")
    output.append(f"Model ID: {exp_meta.get('model_id', 'N/A')}")
    output.append(f"Source Text: {exp_meta.get('source_text_name', 'N/A')}")
    output.append(f"Tokenizer: {exp_meta.get('tokenizer_model', 'N/A')}")
    output.append(f"Embedding Model: {exp_meta.get('embedding_model', 'N/A')}")
    output.append(f"Start Time: {exp_meta.get('start_time', 'N/A')}")
    output.append(f"End Time: {exp_meta.get('end_time', 'N/A')}")
    output.append("")

    # Generation parameters
    output.append("=" * 60)
    output.append("GENERATION PARAMETERS")
    output.append("=" * 60)
    output.append(f"Temperature: {gen_params.get('temperature', 'N/A')}")
    output.append(f"Max Tokens: {gen_params.get('max_tokens', 'N/A')}")
    output.append(f"Top P: {gen_params.get('top_p', 'N/A')}")
    output.append(f"Top K: {gen_params.get('top_k', 'N/A')}")
    output.append(f"Min P: {gen_params.get('min_p', 'N/A')}")
    output.append(f"Repetition Penalty: {gen_params.get('rep_pen', 'N/A')}")
    output.append(f"No Think Mode: {gen_params.get('no_think', 'N/A')}")
    output.append("")

    # Test configuration
    output.append("=" * 60)
    output.append("TEST CONFIGURATION")
    output.append("=" * 60)
    output.append(f"Max Context: {exp_meta.get('max_context', 'N/A')}")
    output.append(f"Start Context: {exp_meta.get('start_context', 'N/A')}")
    output.append(f"Rounds: {exp_meta.get('num_rounds', 'N/A')}")
    output.append(f"Divisions: {exp_meta.get('divisions', 'N/A')}")
    output.append(f"Max Retries: {exp_meta.get('max_retries', 'N/A')}")
    output.append(f"API URL: {exp_meta.get('api_url', 'N/A')}")
    output.append(f"Successful Contexts: {exp_meta.get('successful_contexts', 'N/A')}")
    output.append("")

    # Context lengths tested
    if 'context_lengths' in metadata:
        output.append("=" * 60)
        output.append("CONTEXT LENGTHS TESTED")
        output.append("=" * 60)
        context_lengths = metadata['context_lengths']
        output.append(", ".join([f"{c:,}" for c in context_lengths]))
        output.append("")

    return "\n".join(output)


def _format_nested_dict(data: Any, indent: int = 0) -> List[str]:
    """Recursively format nested dictionary.

    Args:
        data: Dictionary or other data to format
        indent: Current indentation level

    Returns:
        List of formatted lines
    """
    lines = []
    prefix = "  " * indent

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                lines.extend(_format_nested_dict(value, indent + 1))
            elif isinstance(value, list):
                lines.append(f"{prefix}{key}:")
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        lines.append(f"{prefix}  [{i}]:")
                        lines.extend(_format_nested_dict(item, indent + 2))
                    else:
                        lines.append(f"{prefix}  - {item}")
            else:
                lines.append(f"{prefix}{key}: {value}")
    else:
        lines.append(f"{prefix}{data}")

    return lines
