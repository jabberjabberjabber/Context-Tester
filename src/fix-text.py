#!/usr/bin/env python3
"""
Model Readability Degradation Test - Streamlined Architecture

Tests LLM output quality degradation at increasing context lengths by analyzing
the readability complexity of generated text continuations using Cloze scores.

Refactored for maintainability with modular architecture.
"""

import os
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import argparse

from extractous import Extractor


# ================================
# TEXT PROCESSING
# ================================

def normalize_content(content: str) -> str:
    """Convert fixed-width text and normalize characters."""
    content = unicodedata.normalize('NFKC', content)
    content = content.replace('--', '—')
    content = content.replace('"', '"').replace('"', '"')
    content = content.replace(''', "'").replace(''', "'")
    #content = content.replace(' . . . ', "").replace(''', "'")
    text = content.replace('\r\n', '\n').replace('\r', '\n')
    paragraphs = text.split('\n\n')

    result = '\n\n'.join(para.replace('\n', ' ') for para in paragraphs)
    #result = result.replace('\n\n', '\n\n    ')
    return result


def load_reference_text(file_path: str) -> str:
    """Load reference text from file (supports txt, pdf, html)."""
    extractor = Extractor()
    extractor = extractor.set_extract_string_max_length(999999999)
    content, metadata = extractor.extract_file_to_string(file_path)
    print(f"Loaded reference text: {metadata.get('resourceName', file_path)}")
    print(f"Content type: {metadata.get('Content-Type', 'Unknown')}")

    if not content or not content.strip():
        raise ValueError(f"File {file_path} appears to be empty or could not be read")

    normalized_content = normalize_content(content)
    print(f"Total characters: {len(normalized_content):,}")
    print(f"Sample: {normalized_content[150:4000]}")
    return normalized_content




def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="fix text",
        formatter_class=argparse.RawDescriptionHelpFormatter)
            
        # Input file or results directory
    parser.add_argument(
        'input_file',
        nargs='?',
        help='Path to text'
    )
    args = parser.parse_args()    
    
    if not args.input_file:
        print("No input file specified")
        sys.exit(1)
    text = load_reference_text(args.input_file)
    fixed_text = normalize_content(text)
    output_file = os.path.splitext(args.input_file)[0] + "_fixed.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text) 
    


if __name__ == "__main__":
    exit(main())
