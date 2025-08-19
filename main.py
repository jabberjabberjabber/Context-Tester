#!/usr/bin/env python3
"""
Model Readability Degradation Test - Multi-Round Version with Fixed Continuation Point

Tests LLM output quality degradation at increasing context lengths by analyzing
the readability complexity of generated text continuations using Cloze scores.
Now supports multiple test rounds per context length with statistical averaging.

CRITICAL FIX: All continuations now start from the same story position to eliminate
confounding variables. Uses backward-expanding context windows from a fixed point.

Based on observations that models exhibit "flattening" patterns around 8k+ context,
where sentence structure becomes repetitive and vocabulary diversity drops.
"""

import argparse
import json
import math
import re
import warnings
import csv
import statistics
import unicodedata

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Literal, TypeAlias
import requests
from chunker_regex import chunk_regex

from requests.exceptions import RequestException
from bs4 import BeautifulSoup
from extractous import Extractor

# Outlier detection functionality
from outlier_detection import (
    analyze_round_outliers, 
    filter_outliers_from_results,
    print_outlier_summary
)

# Suppress BeautifulSoup warnings
warnings.filterwarnings("ignore", category=UserWarning, module="bs4")


# ================================
# READABILITY ANALYSIS COMPONENTS
# ================================

def load_easy_words(word_list_path: str = "easy_words.txt") -> Set[str]:
    """ Load the Dale-Chall easy words list from file """
    try:
        with open(word_list_path, 'r', encoding='utf-8') as f:
            return {line.strip().lower() for line in f if line.strip()}
    except FileNotFoundError:
        print(f"ERROR: Could not find {word_list_path}")
        print("Please ensure the Dale-Chall word list file exists.")
        raise

EASY_WORDS: Set[str] = set()

def initialize_word_list(word_list_path: str = "easy_words.txt") -> None:
    """ Initialize the global easy words list """
    global EASY_WORDS
    EASY_WORDS = load_easy_words(word_list_path)

def compute_cloze_score(pct_unfamiliar_words: float, avg_sentence_length: float) -> float:
    """ Compute the Cloze score (Chall & Dale, 1995)
    
    Formula: 64 - (95 × pct_unfamiliar_words) - (0.69 × avg_sentence_length)
    """
    raw_result = 64 - (95 * pct_unfamiliar_words) - (0.69 * avg_sentence_length)
    return round(raw_result, 2)

ReadingLevel: TypeAlias = Literal[
    "1", "2", "3", "4", "5-6", "7-8", "9-10", "11-12", "13-15", "16+"
]

class RangeDict(dict[range, ReadingLevel]):
    """ Maps cloze score ranges to reading levels """
    def __getitem__(self, item: Any) -> ReadingLevel:
        int_item = math.ceil(item)
        for key in self.keys():
            if int_item in key:
                return super().__getitem__(key)
        raise KeyError(item)

EQUIV_CLOZE_AND_READING_LEVELS = RangeDict({
    range(58, 65): "1",
    range(54, 58): "2", 
    range(50, 54): "3",
    range(45, 50): "4",
    range(40, 45): "5-6",
    range(34, 40): "7-8", 
    range(28, 34): "9-10",
    range(22, 28): "11-12",
    range(16, 22): "13-15",
    range(10, 16): "16+",
})

def reading_level_from_cloze(cloze_score: float) -> ReadingLevel:
    """ Convert cloze score to reading level """
    bounded_score = max(10, min(64, cloze_score))
    return EQUIV_CLOZE_AND_READING_LEVELS[bounded_score]

def pct_unfamiliar_words(text: str) -> float:
    """ Calculate percentage of unfamiliar words """
    words = _words(text)
    if not words:
        return 0.0
    
    no_possessives = (w.replace("'s", "").replace("s'", "") for w in words)
    unfamiliar_words = [w for w in no_possessives if _is_unfamiliar(w)]
    return len(unfamiliar_words) / len(words)

def avg_sentence_length(text: str) -> float:
    """ Calculate average sentence length in words """
    cleaned_text = text.replace("\n", " ").strip()
    sentences = re.findall(r"\b[^.!?]+[.!?]*", cleaned_text, re.UNICODE)
    words = _words(text)
    
    if not sentences:
        return 0.0
    return len(words) / len(sentences)

def _words(in_text: str) -> tuple[str, ...]:
    """ Extract normalized words from text """
    plain_text = BeautifulSoup(in_text, "html.parser").text
    return tuple(w.lower().strip('.(),"\'') for w in plain_text.split() if w.strip())

def _is_unfamiliar(word: str) -> bool:
    """ Check if word is unfamiliar (not in Dale-Chall list) """
    if word.isdigit():
        return False
    return word not in EASY_WORDS

def is_power_of_two(n: int) -> bool:
    """Check if n is a power of 2"""
    return n > 0 and (n & (n - 1)) == 0

def sentence_length_variance(text: str) -> float:
    """ Calculate variance in sentence lengths """
    cleaned_text = text.replace("\n", " ").strip()
    sentences = re.findall(r"\b[^.!?]+[.!?]*", cleaned_text, re.UNICODE)
    
    if len(sentences) < 2:
        return 0.0
        
    lengths = [len(sentence.split()) for sentence in sentences]
    return statistics.variance(lengths) if len(lengths) > 1 else 0.0

def analyze_text_readability(text: str) -> Dict[str, Any]:
    """ Complete readability analysis """
    if not text.strip():
        return {
            'pct_unfamiliar_words': 0.0,
            'avg_sentence_length': 0.0,
            'cloze_score': 64.0,
            'reading_level': "1",
            'word_count': 0,
            'sentence_count': 0,
            'sentence_length_variance': 0.0,
            'vocabulary_diversity': 0.0
        }
    
    words = _words(text)
    pct_unfamiliar = pct_unfamiliar_words(text)
    avg_sent_len = avg_sentence_length(text)
    cloze_score = compute_cloze_score(pct_unfamiliar, avg_sent_len)
    reading_level = reading_level_from_cloze(cloze_score)
    
    # Additional metrics for degradation detection
    unique_words = set(words)
    vocab_diversity = len(unique_words) / len(words) if words else 0.0
    
    cleaned_text = text.replace("\n", " ").strip()
    sentences = re.findall(r"\b[^.!?]+[.!?]*", cleaned_text, re.UNICODE)
    
    return {
        'pct_unfamiliar_words': round(pct_unfamiliar, 4),
        'avg_sentence_length': round(avg_sent_len, 2),
        'cloze_score': cloze_score,
        'reading_level': reading_level,
        'word_count': len(words),
        'sentence_count': len(sentences),
        'sentence_length_variance': round(sentence_length_variance(text), 2),
        'vocabulary_diversity': round(vocab_diversity, 4)
    }


# ================================
# API CLIENT FOR TEXT GENERATION
# ================================

class StreamingAPIClient:
    """ Client for generating text continuations via streaming API """
    
    def __init__(self, api_url: str, api_password: Optional[str] = None):
        self.api_url = api_url
        if not self.api_url.endswith('/v1/chat/completions'):
            self.api_url = f"{self.api_url.rstrip('/')}/v1/chat/completions"
            
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        
        if api_password:
            self.headers["Authorization"] = f"Bearer {api_password}"
            
    def count_tokens(self, text: str) -> int:
        base_url = self.api_url.replace('v1/chat/completions', '')
        try:
            response = requests.post(
                f"{base_url}/api/extra/tokencount",
                json={"prompt": text},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if "value" in data:
                    token_count = data["value"]
                return token_count
                
            return None
        except Exception as e:
            print(f"Error counting tokens ({e})")
            return None
        
            
    def prune_text(self, text: str, max_context: int = 32768):
        
        """ Get max amount of text that fits into a natural breakpoint
        """
        
        total_tokens = self.count_tokens(text)
        if total_tokens < max_context:
            return text

        # chunk_regex is designed to break at natural language points
        # to preserve context and readability
        matches = chunk_regex.finditer(text)
        current_size = 0
        chunks = []
        
        for match in matches:
            chunk = match.group(0)
            chunk_size = self.count_tokens(chunk)
            if current_size + chunk_size > (max_context * 0.9):
                if not chunks:
                    chunks.append(chunk)
                break
            chunks.append(chunk)
            current_size += chunk_size
        
        return ''.join(chunks)
        
    def tokenize_text_batched(self, text: str, chunk_size: int = 45000) -> List[int]:
        """ Tokenize large text by batching API calls, return token IDs """
        if not text or not text.strip():
            return []
        
        base_url = self.api_url.replace('/v1/chat/completions', '')
        all_token_ids = []
        
        # Split text into manageable chunks
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        print(f"Tokenizing text in {len(chunks)} chunks...")
        
        for i, chunk in enumerate(chunks, 1):
            try:
                response = requests.post(
                    f"{base_url}/api/extra/tokencount",
                    json={"prompt": chunk},
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if "ids" in data:
                        chunk_tokens = data["ids"]
                        all_token_ids.extend(chunk_tokens)
                        print(f"  Chunk {i}/{len(chunks)}: {len(chunk_tokens)} tokens")
                    else:
                        print(f"  Chunk {i}/{len(chunks)}: No token IDs returned")
                        # Fallback estimation
                        estimated_tokens = int(len(chunk.split()) * 1.33)
                        all_token_ids.extend(range(len(all_token_ids), len(all_token_ids) + estimated_tokens))
                else:
                    print(f"  Chunk {i}/{len(chunks)}: API error {response.status_code}")
                    # Fallback estimation
                    estimated_tokens = int(len(chunk.split()) * 1.33)
                    all_token_ids.extend(range(len(all_token_ids), len(all_token_ids) + estimated_tokens))
                    
            except Exception as e:
                print(f"  Chunk {i}/{len(chunks)}: Error {e}")
                # Fallback estimation
                estimated_tokens = int(len(chunk.split()) * 1.33)
                all_token_ids.extend(range(len(all_token_ids), len(all_token_ids) + estimated_tokens))
        
        print(f"Total tokens collected: {len(all_token_ids):,}")
        return all_token_ids
    
    def tokens_to_text(self, token_ids: List[int]) -> str:
        """ Convert token IDs back to text via API """
        if not token_ids:
            return ""
        
        try:
            base_url = self.api_url.replace('/v1/chat/completions', '')
            response = requests.post(
                f"{base_url}/api/extra/detokenize",
                json={"ids": token_ids},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success", False):
                    return data.get("result", "")
                else:
                    print(f"Token detokenize failed: success=False")
                    return ""
            else:
                print(f"Token detokenize failed: {response.status_code}")
                return ""
                
        except Exception as e:
            print(f"Token detokenize error: {e}")
            return ""
    
    def get_max_context_length(self) -> int:
        """ Get model's maximum context length from API """
        try:
            base_url = self.api_url.replace('/v1/chat/completions', '')
            response = requests.get(
                f"{base_url}/api/extra/true_max_context_length", 
                timeout=10
            )
            if response.status_code == 200:
                max_context = int(response.json().get("value", 32768))
                print(f"Detected model max context: {max_context:,}")
                return max_context
            else:
                print(f"Could not detect max context, using default: 32768")
                return 32768
        except Exception as e:
            print(f"Error detecting max context ({e}), using default: 32768")
            return 32768
    
    def get_model_name(self) -> Optional[str]:
        """ Get model name from API """
        try:
            base_url = self.api_url.replace('/v1/chat/completions', '')
            response = requests.get(
                f"{base_url}/api/v1/model",
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                if "result" in data:
                    model_name = str(data["result"]).replace('koboldcpp/', '')
                    print(f"Detected model name: {model_name}")
                    return model_name
            print("Could not detect model name from API")
            return None
        except Exception as e:
            print(f"Error detecting model name ({e})")
            return None
    
    def generate_continuation(self, context: str, max_tokens: int = 1024,
                            temperature: float = 1.0, top_k: int = 100, 
                            top_p: float = 1.0, min_p: float = 0.1, 
                            rep_pen: float = 1.01) -> str:
        """ Generate text continuation from context """
        
        instruction = """Continue this story for as long as you can. Do not try to add a conclusion or ending, just keep writing as if this were part of the middle of a novel. Maintain the same style, tone, and narrative voice. Focus on developing the plot, characters, and setting naturally."""
        print(f"Starting from: {context[-20:]}")
        payload = {
            "messages": [
                {"role": "system", "content": "You are a skilled novelist continuing a story."},
                {"role": "user", "content": f"{context}\n\n{instruction}"}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": rep_pen,
            "top_k": top_k,
            "stream": True,
            "min_p": min_p
        }
        
        result = []
        
        try:
            response = requests.post(
                self.api_url,
                json=payload,
                headers=self.headers,
                stream=True,
                timeout=300
            )
            
            for line in response.iter_lines():
                if not line:
                    continue
                    
                line_text = line.decode('utf-8')
                if line_text.startswith('data: '):
                    line_text = line_text[6:]
                
                if line_text == '[DONE]':
                    break
                    
                try:
                    data = json.loads(line_text)
                    if 'choices' in data and len(data['choices']) > 0:
                        if 'delta' in data['choices'][0]:
                            if 'content' in data['choices'][0]['delta']:
                                token = data['choices'][0]['delta']['content']
                                result.append(token)
                                print(token, end='', flush=True)
                except json.JSONDecodeError:
                    continue
            
            print()  # New line after generation
            return ''.join(result)
                
        except Exception as e:
            print(f"\nError in generation: {str(e)}")
            return ""


# ================================
# MAIN DEGRADATION TESTER
# ================================

class ReadabilityDegradationTester:
    """ Tests model degradation across increasing context lengths with fixed continuation point """
    
    def __init__(self, api_url: str, api_password: Optional[str] = None,
                 word_list_path: str = "easy_words.txt", num_rounds: int = 1, divisions: int = 1,
                 model_name: Optional[str] = None, max_tokens: int = 1024, 
                 temperature: float = 1.0, top_k: int = 100, top_p: float = 1.0, 
                 min_p: float = 0.1, rep_pen: float = 1.01):
        """ Initialize the degradation tester
        
        Args:
            api_url: URL to the LLM API
            api_password: Optional API key
            word_list_path: Path to Dale-Chall word list
            num_rounds: Number of test rounds per context length
            divisions: Number of splits per power of 2 tier
            model_name: Optional model name override
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
            top_k: Top-k sampling
            top_p: Top-p sampling  
            min_p: Min-p sampling
            rep_pen: Repetition penalty
        """
        self.client = StreamingAPIClient(api_url, api_password)
        initialize_word_list(word_list_path)
        self.num_rounds = max(1, num_rounds)
        self.results = []
        self.detailed_results = []  # Store individual round results
        self.divisions = max(1, divisions)
        self.all_tokens = []  # Full tokenized text
        self.working_tokens = []  # Last max_context tokens
        
        # Generation parameters
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p
        self.rep_pen = rep_pen
        
        # Model name (try API first, then fallback to provided name)
        self.model_name = model_name or self.client.get_model_name() or "unknown-model"
    
    def generate_output_filename(self) -> str:
        """ Generate output filename based on model name and test parameters """
        # Clean model name for filename
        clean_model = self.model_name.replace('/', '-').replace('\\', '-').replace(':', '-')
        
        # Generate filename with key parameters
        filename = f"{clean_model}-{self.num_rounds}rounds_{self.divisions}divs.csv"
        return filename
        
    def normalize_content(self, content: str) -> str:
        """ Convert fixed-width text and normalize characters """
        content = unicodedata.normalize('NFKC', content)
        content = content.replace('--', '—')
        paragraphs = content.split('\n\n')
        reformatted_paragraphs = []
        
        for paragraph in paragraphs:
            # Skip empty paragraphs
            if not paragraph.strip():
                continue
                
            # Join lines within paragraph, collapsing artificial line breaks
            lines = paragraph.strip().split('\n')
            joined = ' '.join(line.strip() for line in lines if line.strip())
            
            # Clean up multiple spaces
            cleaned = ' '.join(joined.split())
            
            if cleaned:
                reformatted_paragraphs.append(cleaned)
        
        # Rejoin with double newlines to preserve paragraph structure
        return '\n\n'.join(reformatted_paragraphs)
        
    def load_reference_text(self, file_path: str) -> str:
        """ Load reference text from file """
        extractor = Extractor()
        try:
            content, metadata = extractor.extract_file_to_string(file_path)
            print(f"Loaded reference text: {metadata.get('resourceName', file_path)}")
            print(f"Content type: {metadata.get('Content-Type', 'Unknown')}")
            
            # Debug text loading
            if not content or not content.strip():
                raise ValueError(f"File {file_path} appears to be empty or could not be read")
            
            normalized_content = self.normalize_content(content)
            print(f"Text preview (first 200 chars): {normalized_content[:200]!r}")
            print(f"Total characters: {len(normalized_content):,}")
            
            return normalized_content
        except Exception as e:
            print(f"Error loading file: {e}")
            # Try simple file reading as fallback
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():
                        normalized_content = self.normalize_content(content)
                        print(f"Fallback: Successfully loaded {len(normalized_content):,} characters")
                        return normalized_content
            except Exception as e2:
                print(f"Fallback reading also failed: {e2}")
            raise
    
    def prepare_tokenized_text(self, text: str, max_context: int) -> bool:
        """ Tokenize text and prepare working token set 
        
        Args:
            text: Reference text to tokenize
            max_context: Maximum context length we'll test
            
        Returns:
            True if successful, False if insufficient text
        """
        print(f"\n{'='*60}")
        print("PREPARING TOKENIZED TEXT")
        print(f"{'='*60}")
        
        # Prune the text to a natural breakpoint
        pruned_text = self.client.prune_text(text, max_context)
        
        # Tokenize the full text
        self.all_tokens = self.client.tokenize_text_batched(pruned_text)
        
        if not self.all_tokens:
            print("ERROR: Failed to tokenize text")
            return False
        
        total_tokens = len(self.all_tokens)
        print(f"Total tokens in reference text: {total_tokens:,}")
        
        # Check if we have enough tokens for testing
        min_required = int(max_context * 0.9)
        if total_tokens < min_required:
            print(f"ERROR: Insufficient text. Need at least {min_required:,} tokens, got {total_tokens:,}")
            return False
        
        # Take the last max_context tokens as our working set
        self.working_tokens = self.all_tokens[-max_context:]
        print(f"Working with last {len(self.working_tokens):,} tokens for testing")
        
        return True
    
    def generate_context_lengths(self, max_context: int) -> List[int]:
        """ Generate power-of-2 context lengths to test with subdivisions
        
        Args:
            max_context: Maximum context length
            
        Returns:
            List of context lengths in tokens
        """
        # Generate base powers of 2: 1k, 2k, 4k, 8k, 16k, 32k, etc.
        tiers = []
        power = 10  # Start at 2^10 = 1024
        
        while True:
            length = 2 ** power
            if length > max_context:
                break
            tiers.append(length)
            power += 1
        
        if not tiers:
            # Fallback for very small max_context
            return [min(1024, max_context)]
        
        # Apply subdivision logic
        context_lengths = [tiers[0]]
        
        for i in range(len(tiers) - 1):
            start, end = tiers[i], tiers[i + 1]
            step = (end - start) / self.divisions
            
            # Insert divisions-1 intermediate values
            for j in range(1, self.divisions):
                context_lengths.append(int(start + step * j))
            
            context_lengths.append(end)
        
        print(f"Generated context lengths: {context_lengths}")
        return context_lengths
    
    def build_context_window(self, context_length: int) -> str:
        """ Build context window from working tokens using backward expansion
        
        Args:
            context_length: Number of tokens to include in context
            
        Returns:
            Context text for this tier
        """
        if context_length > len(self.working_tokens):
            # Use all available working tokens
            context_tokens = self.working_tokens
        else:
            # Take the last context_length tokens from working set
            context_tokens = self.working_tokens[-context_length:]
        
        # Convert tokens back to text
        context_text = self.client.tokens_to_text(context_tokens)
        return context_text
    
    def average_results(self, round_results: List[Dict]) -> Dict[str, Any]:
        """ Average results across multiple rounds """
        if not round_results:
            return {}
        
        if len(round_results) == 1:
            result = round_results[0].copy()
            # Add std fields for single results
            result['num_rounds'] = 1
            result['cloze_score_std'] = 0.0
            result['vocab_diversity_std'] = 0.0
            result['timestamp'] = datetime.now().isoformat()
            return result
        
        # Numerical fields to average
        numerical_fields = [
            'pct_unfamiliar_words', 'avg_sentence_length', 'cloze_score',
            'word_count', 'sentence_count', 'sentence_length_variance',
            'vocabulary_diversity', 'continuation_length'
        ]
        
        averaged = {}
        
        # Copy non-numerical fields from first result
        for key, value in round_results[0].items():
            if key not in numerical_fields:
                averaged[key] = value
        
        # Average numerical fields
        for field in numerical_fields:
            values = [result[field] for result in round_results if field in result and result[field] is not None]
            if values:
                averaged[field] = round(statistics.mean(values), 4)
            else:
                averaged[field] = 0.0
        
        # Recalculate reading level from averaged cloze score
        if 'cloze_score' in averaged:
            averaged['reading_level'] = reading_level_from_cloze(averaged['cloze_score'])
        
        # Add round statistics
        averaged['num_rounds'] = len(round_results)
        
        # Calculate std deviations safely
        cloze_values = [r['cloze_score'] for r in round_results if 'cloze_score' in r and r['cloze_score'] is not None]
        vocab_values = [r['vocabulary_diversity'] for r in round_results if 'vocabulary_diversity' in r and r['vocabulary_diversity'] is not None]
        
        averaged['cloze_score_std'] = round(statistics.stdev(cloze_values), 3) if len(cloze_values) > 1 else 0.0
        averaged['vocab_diversity_std'] = round(statistics.stdev(vocab_values), 4) if len(vocab_values) > 1 else 0.0
        
        # Update timestamp to when averaging was done
        averaged['timestamp'] = datetime.now().isoformat()
        
        return averaged
    
    def run_test(self, file_path: str, max_context: Optional[int] = None) -> List[Dict]:
        """ Run the complete degradation test with fixed continuation point
        
        Args:
            file_path: Path to reference text file
            max_context: Maximum context length to test
            
        Returns:
            List of averaged test results
        """
        print("=" * 60)
        print("MODEL READABILITY DEGRADATION TEST - FIXED CONTINUATION POINT")
        if self.num_rounds > 1:
            print(f"Running {self.num_rounds} rounds per context length")
        print("=" * 60)
        
        # Load reference text
        reference_text = self.load_reference_text(file_path)
        
        # Determine max context
        if max_context is None:
            max_context = self.client.get_max_context_length()
        else:
            print(f"Using user-specified max context: {max_context:,}")
        
        # Prepare tokenized text
        if not self.prepare_tokenized_text(reference_text, max_context):
            print("Failed to prepare tokenized text")
            return []
        
        # Analyze baseline readability
        baseline_text = self.client.tokens_to_text(self.working_tokens[-5000:] if len(self.working_tokens) > 5000 else self.working_tokens)
        baseline_analysis = analyze_text_readability(baseline_text)
        print(f"\nBaseline text readability (from continuation point area):")
        print(f"  Cloze Score: {baseline_analysis['cloze_score']}")
        print(f"  Reading Level: {baseline_analysis['reading_level']}")
        print(f"  Vocabulary Diversity: {baseline_analysis['vocabulary_diversity']}")
        
        # Generate test context lengths
        context_lengths = self.generate_context_lengths(max_context)
        
        print(f"\n{'='*60}")
        print("RUNNING DEGRADATION TESTS")
        print(f"All continuations start from same story position!")
        print(f"{'='*60}")
        
        self.results = []
        self.detailed_results = []
        
        for i, context_length in enumerate(context_lengths, 1):
            print(f"\n[TEST {i}/{len(context_lengths)}] Context Length: {context_length:,} tokens")
            if self.num_rounds > 1:
                print(f"Running {self.num_rounds} rounds...")
            print("-" * 50)
            
            # Build context window (same for all rounds)
            context = self.build_context_window(context_length)
            
            if not context:
                print(f"WARNING: Failed to build context for {context_length} tokens")
                continue
            
            print(f"Context built successfully ({len(context.split())} words)")
            
            # Run multiple rounds
            round_results = []
            
            for round_num in range(self.num_rounds):
                if self.num_rounds > 1:
                    print(f"\n  Round {round_num + 1}/{self.num_rounds}:")
                else:
                    print(f"Generating continuation...")
                
                # Generate continuation
                continuation = self.client.generate_continuation(
                    context, 
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    min_p=self.min_p,
                    rep_pen=self.rep_pen
                )
                
                if not continuation:
                    print(f"WARNING: No continuation generated for round {round_num + 1}")
                    continue
                
                # Analyze readability
                analysis = analyze_text_readability(continuation)
                
                # Store detailed result
                detailed_result = {
                    'context_length': context_length,
                    'actual_context_tokens': context_length,  # Now exact
                    'round_number': round_num + 1,
                    'continuation_length': len(continuation),
                    'timestamp': datetime.now().isoformat(),
                    **analysis
                }
                
                round_results.append(detailed_result)
                self.detailed_results.append(detailed_result)
                
                # Display round results
                if self.num_rounds > 1:
                    print(f"    Cloze: {analysis['cloze_score']:6.2f}, "
                          f"Level: {analysis['reading_level']:>6}, "
                          f"Vocab: {analysis['vocabulary_diversity']:5.3f}")
            
            if not round_results:
                print(f"WARNING: No successful rounds for context length {context_length}")
                continue
            
            # Average results across rounds
            averaged_result = self.average_results(round_results)
            self.results.append(averaged_result)
            
            # Display averaged results
            print(f"\nAveraged Results (n={len(round_results)}):")
            print(f"  Cloze Score: {averaged_result['cloze_score']:6.2f} "
                  f"(±{averaged_result['cloze_score_std']:4.2f})")
            print(f"  Reading Level: {averaged_result['reading_level']:>6}")
            print(f"  Unfamiliar Words: {averaged_result['pct_unfamiliar_words']*100:5.1f}%")
            print(f"  Avg Sentence Length: {averaged_result['avg_sentence_length']:5.1f}")
            print(f"  Sentence Variance: {averaged_result['sentence_length_variance']:5.1f}")
            print(f"  Vocabulary Diversity: {averaged_result['vocabulary_diversity']:5.3f} "
                  f"(±{averaged_result['vocab_diversity_std']:5.3f})")
        
        # Outlier analysis and filtering
        if len(self.detailed_results) >= 4:  # Need minimum data for outlier detection
            try:
                outlier_analysis = analyze_round_outliers(self.detailed_results)
                if outlier_analysis.get('has_outliers', False):
                    print_outlier_summary(outlier_analysis, max_context)
                    # Filter severe outliers from detailed results for final averaging
                    filtered_results = filter_outliers_from_results(
                        self.detailed_results, outlier_analysis, exclude_severe_only=True
                    )
                    if len(filtered_results) != len(self.detailed_results):
                        print(f"\n📊 Filtered {len(self.detailed_results) - len(filtered_results)} severe outlier rounds")
                        # Recalculate averaged results without severe outliers
                        context_groups = {}
                        for result in filtered_results:
                            ctx_len = result['context_length']
                            if ctx_len not in context_groups:
                                context_groups[ctx_len] = []
                            context_groups[ctx_len].append(result)
                        
                        # Rebuild results with filtered data
                        filtered_averaged_results = []
                        for ctx_len in sorted(context_groups.keys()):
                            if context_groups[ctx_len]:  # Make sure we have data
                                averaged = self.average_results(context_groups[ctx_len])
                                filtered_averaged_results.append(averaged)
                        
                        if filtered_averaged_results:  # Only update if we got valid results
                            self.results = filtered_averaged_results
            except Exception as e:
                print(f"\n⚠️  Outlier analysis failed: {e}")
                print("Continuing with original results...")
        
        # Analysis and summary
        self._print_summary()
        
        # Auto-generate filename and save results
        output_file = self.generate_output_filename()
        self._save_results(output_file)
        
        return self.results
    
    def _print_summary(self):
        """ Print test summary and degradation analysis """
        if len(self.results) < 2:
            return
        
        print(f"\n{'='*60}")
        print("DEGRADATION ANALYSIS")
        if self.num_rounds > 1:
            print(f"(Based on averages of {self.num_rounds} rounds each)")
        print("✅ All continuations from SAME story position")
        print(f"{'='*60}")
        
        # Calculate trends
        cloze_scores = [r['cloze_score'] for r in self.results]
        vocab_diversity = [r['vocabulary_diversity'] for r in self.results]
        sentence_variance = [r['sentence_length_variance'] for r in self.results]
        
        print(f"\nCloze Score Trend:")
        print(f"  Starting: {cloze_scores[0]:6.2f}")
        print(f"  Ending:   {cloze_scores[-1]:6.2f}")
        change = cloze_scores[-1] - cloze_scores[0]
        print(f"  Change:   {change:+6.2f} {'(simpler text)' if change > 0 else '(more complex text)'}")
        
        print(f"\nVocabulary Diversity Trend:")
        print(f"  Starting: {vocab_diversity[0]:6.3f}")
        print(f"  Ending:   {vocab_diversity[-1]:6.3f}")
        vocab_change = vocab_diversity[-1] - vocab_diversity[0]
        print(f"  Change:   {vocab_change:+6.3f} {'(more repetitive)' if vocab_change < 0 else '(more diverse)'}")
        
        # Look for degradation points
        degradation_points = []
        
        for i in range(1, len(self.results)):
            prev_score = cloze_scores[i-1]
            curr_score = cloze_scores[i]
            prev_vocab = vocab_diversity[i-1]
            curr_vocab = vocab_diversity[i]
            prev_variance = sentence_variance[i-1]
            curr_variance = sentence_variance[i]
            
            # Significant RISE in cloze score = simplification = degradation
            if curr_score - prev_score > 3.0:
                degradation_points.append({
                    'context_length': self.results[i]['context_length'],
                    'metric': 'cloze_score',
                    'change': curr_score - prev_score,
                    'direction': 'rose'
                })
            
            # Significant DROP in vocabulary diversity = degradation
            if prev_vocab - curr_vocab > 0.05:
                degradation_points.append({
                    'context_length': self.results[i]['context_length'],
                    'metric': 'vocabulary_diversity',
                    'change': prev_vocab - curr_vocab,
                    'direction': 'dropped'
                })
            
            # Significant DROP in sentence variance = degradation
            if prev_variance - curr_variance > 5.0:
                degradation_points.append({
                    'context_length': self.results[i]['context_length'],
                    'metric': 'sentence_variance',
                    'change': prev_variance - curr_variance,
                    'direction': 'dropped'
                })
        
        if degradation_points:
            print(f"\n⚠️ DEGRADATION DETECTED:")
            for point in degradation_points:
                print(f"  At {point['context_length']:,} tokens: {point['metric']} {point['direction']} {point['change']:.3f}")
        else:
            print(f"\n✅ No significant degradation detected in tested range")
        
        # Print full results table
        print(f"\nFULL RESULTS:")
        if self.num_rounds > 1:
            print(f"{'Context':>8} {'Cloze':>8} {'±Std':>6} {'Level':>8} {'Unfamiliar':>10} {'AvgSent':>8} {'Variance':>8} {'VocabDiv':>8} {'±Std':>6}")
            print("-" * 80)
            
            for result in self.results:
                print(f"{result['context_length']:>8,} "
                      f"{result['cloze_score']:>8.2f} "
                      f"{result['cloze_score_std']:>6.2f} "
                      f"{result['reading_level']:>8} "
                      f"{result['pct_unfamiliar_words']*100:>9.1f}% "
                      f"{result['avg_sentence_length']:>8.1f} "
                      f"{result['sentence_length_variance']:>8.1f} "
                      f"{result['vocabulary_diversity']:>8.3f} "
                      f"{result['vocab_diversity_std']:>6.3f}")
        else:
            print(f"{'Context':>8} {'Cloze':>8} {'Level':>8} {'Unfamiliar':>10} {'AvgSent':>8} {'Variance':>8} {'VocabDiv':>8}")
            print("-" * 68)
            
            for result in self.results:
                print(f"{result['context_length']:>8,} "
                      f"{result['cloze_score']:>8.2f} "
                      f"{result['reading_level']:>8} "
                      f"{result['pct_unfamiliar_words']*100:>9.1f}% "
                      f"{result['avg_sentence_length']:>8.1f} "
                      f"{result['sentence_length_variance']:>8.1f} "
                      f"{result['vocabulary_diversity']:>8.3f}")
    
    def _save_results(self, output_file: str):
        """ Save results to CSV """
        if not self.results:
            return
        
        # Combine averaged results with model and generation info
        enhanced_results = []
        for result in self.results:
            enhanced_result = {
                'model_name': self.model_name,
                'max_tokens': self.max_tokens,
                'temperature': self.temperature,
                'top_k': self.top_k,
                'top_p': self.top_p,
                'min_p': self.min_p,
                'rep_pen': self.rep_pen,
                **result
            }
            enhanced_results.append(enhanced_result)
        
        # Save enhanced results
        fieldnames = enhanced_results[0].keys()
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(enhanced_results)
        
        print(f"\n📊 Results saved to: {output_file}")
        print(f"   Model: {self.model_name}")
        print(f"   Rounds: {self.num_rounds}, Divisions: {self.divisions}")
        print(f"   Generation params: max_tokens={self.max_tokens}, temp={self.temperature}, top_k={self.top_k}")
        print(f"                     top_p={self.top_p}, min_p={self.min_p}, rep_pen={self.rep_pen}")


# ================================
# COMMAND LINE INTERFACE
# ================================

def main():
    parser = argparse.ArgumentParser(
        description="Test LLM readability degradation across context lengths with fixed continuation point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py novel.txt --api-url http://localhost:5001
  python main.py document.pdf --max-context 16384 --model-name "MyModel"
  python main.py text.txt --word-list dale_chall_words.txt --rounds 3
  python main.py novel.txt --rounds 5 --divisions 2 --temp 0.8 --top-k 50
  python main.py text.txt --max-tokens 1024 --rep-pen 1.05 --min-p 0.05

CRITICAL IMPROVEMENT: All continuations now start from the same story position!
This eliminates confounding variables from different story contexts.

Output files are automatically named based on model and test parameters.
        """
    )
    
    parser.add_argument(
        'input_file',
        help='Path to reference text file (any format supported by extractous)'
    )
    
    parser.add_argument(
        '--api-url',
        default='http://localhost:5001',
        help='API URL for the LLM service'
    )
    
    parser.add_argument(
        '--api-password',
        default=None,
        help='API key/password if required'
    )
    
    parser.add_argument(
        '--word-list',
        default='easy_words.txt',
        help='Path to Dale-Chall easy words list'
    )
    
    parser.add_argument(
        '--max-context',
        type=int,
        default=None,
        help='Maximum context length to test (auto-detect if not specified)'
    )
    
    parser.add_argument(
        '--rounds',
        type=int,
        default=3,
        help='Number of test rounds per context length (default: 3)'
    )
    
    parser.add_argument(
        '--divisions',
        type=int,
        default=1,
        help='Number of context divisions between tiers as a power of 2'
    )
    
    parser.add_argument(
        '--model-name',
        default=None,
        help='Override model name (auto-detected if not provided)'
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=512,
        help='Maximum tokens to generate (default: 512)'
    )
    
    parser.add_argument(
        '--temp',
        type=float,
        default=1.0,
        help='Generation temperature (default: 1.0)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=100,
        help='Top-k sampling (default: 100)'
    )
    
    parser.add_argument(
        '--top-p',
        type=float,
        default=1.0,
        help='Top-p sampling (default: 1.0)'
    )
    
    parser.add_argument(
        '--min-p',
        type=float,
        default=0.1,
        help='Min-p sampling (default: 0.1)'
    )
    
    parser.add_argument(
        '--rep-pen',
        type=float,
        default=1.01,
        help='Repetition penalty (default: 1.01)'
    )
    args = parser.parse_args()
    
    if not is_power_of_two(args.divisions):
        print(f"Divisions must be 1 or a power of 2 such as 2 or 4 or 8")
        return 1
        
    try:
        tester = ReadabilityDegradationTester(
            api_url=args.api_url,
            api_password=args.api_password,
            word_list_path=args.word_list,
            num_rounds=args.rounds,
            divisions=args.divisions,
            model_name=args.model_name,
            max_tokens=args.max_tokens,
            temperature=args.temp,
            top_k=args.top_k,
            top_p=args.top_p,
            min_p=args.min_p,
            rep_pen=args.rep_pen
        )
        
        tester.run_test(
            file_path=args.input_file,
            max_context=args.max_context
        )
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())