#!/usr/bin/env python3
"""
Model Readability Degradation Test

Tests LLM output quality degradation at increasing context lengths by analyzing
the readability complexity of generated text continuations using Cloze scores.

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
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Literal, TypeAlias
import requests
from requests.exceptions import RequestException
from bs4 import BeautifulSoup
from extractous import Extractor

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
        """ Count tokens using the API """
        if not text or not text.strip():
            return 0
            
        try:
            base_url = self.api_url.replace('/v1/chat/completions', '')
            response = requests.post(
                f"{base_url}/api/extra/tokencount",
                json={"prompt": text},
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            if response.status_code == 200:
                token_count = int(response.json().get("value", 0))
                if token_count > 0:
                    return token_count
        except Exception as e:
            print(f"Token counting API failed: {e}")
        
        # Robust fallback estimation
        words = text.split()
        if not words:
            return 0
        # More accurate token estimation: ~1.33 tokens per word for English
        estimated = int(len(words) * 1.33)
        print(f"Using estimated token count: {estimated:,} (from {len(words):,} words)")
        return estimated
    
    def generate_continuation(self, context: str, max_tokens: int = 1024,
                            temperature: float = 0.7) -> str:
        """ Generate text continuation from context """
        
        instruction = """Continue this story for as long as you can. Do not try to add a conclusion or ending, just keep writing as if this were part of the middle of a novel. Maintain the same style, tone, and narrative voice. Focus on developing the plot, characters, and setting naturally."""
        
        payload = {
            "messages": [
                {"role": "system", "content": "You are a skilled novelist continuing a story."},
                {"role": "user", "content": f"{context}\n\n{instruction}"}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.95,
            "repetition_penalty": 1.05,
            "stream": True
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
    """ Tests model degradation across increasing context lengths """
    
    def __init__(self, api_url: str, api_password: Optional[str] = None,
                 word_list_path: str = "easy_words.txt"):
        """ Initialize the degradation tester
        
        Args:
            api_url: URL to the LLM API
            api_password: Optional API key
            word_list_path: Path to Dale-Chall word list
        """
        self.client = StreamingAPIClient(api_url, api_password)
        initialize_word_list(word_list_path)
        self.results = []
        
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
            
            print(f"Text preview (first 200 chars): {content[:200]!r}")
            print(f"Total characters: {len(content):,}")
            
            return content
        except Exception as e:
            print(f"Error loading file: {e}")
            # Try simple file reading as fallback
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():
                        print(f"Fallback: Successfully loaded {len(content):,} characters")
                        return content
            except Exception as e2:
                print(f"Fallback reading also failed: {e2}")
            raise
    
    def generate_context_lengths(self, text: str, max_context: Optional[int] = None) -> List[int]:
        """ Generate power-of-2 context lengths to test
        
        Args:
            text: Reference text
            max_context: Maximum context length (default: auto-detect)
            
        Returns:
            List of context lengths in tokens
        """
        if not text or not text.strip():
            print("ERROR: Empty text provided to generate_context_lengths")
            return []
            
        text_tokens = self.client.count_tokens(text)
        print(f"Reference text token count: {text_tokens:,}")
        
        if text_tokens == 0:
            print("ERROR: Could not count tokens in reference text")
            return []
        
        if max_context is None:
            # Try to detect model's max context
            try:
                base_url = self.client.api_url.replace('/v1/chat/completions', '')
                response = requests.get(f"{base_url}/api/extra/true_max_context_length", timeout=10)
                if response.status_code == 200:
                    max_context = int(response.json().get("value", 32768))
                    print(f"Detected model max context: {max_context:,}")
                else:
                    max_context = 32768
                    print(f"Could not detect max context, using default: {max_context:,}")
            except Exception as e:
                max_context = 32768
                print(f"Error detecting max context ({e}), using default: {max_context:,}")
        
        # Generate powers of 2: 1k, 2k, 4k, 8k, 16k, 32k, etc.
        context_lengths = []
        power = 10  # Start at 2^10 = 1024
        
        while True:
            length = 2 ** power
            if length > min(text_tokens, max_context):
                break
            context_lengths.append(length)
            power += 1
        
        # Ensure we have at least one test case
        if not context_lengths and text_tokens > 0:
            # Add smaller lengths if text is very short
            for test_length in [512, 256, 128]:
                if test_length <= text_tokens:
                    context_lengths = [test_length]
                    break
        
        print(f"Generated context lengths: {context_lengths}")
        
        return context_lengths
    
    def truncate_to_tokens(self, text: str, target_tokens: int) -> str:
        """ Truncate text to approximately target token count """
        words = text.split()
        # Rough estimation: 1.3 words per token
        target_words = int(target_tokens / 1.3)
        
        if target_words >= len(words):
            return text
            
        truncated = ' '.join(words[:target_words])
        
        # Fine-tune by checking actual token count
        actual_tokens = self.client.count_tokens(truncated)
        
        # Adjust if needed
        while actual_tokens > target_tokens and target_words > 0:
            target_words = int(target_words * 0.9)
            truncated = ' '.join(words[:target_words])
            actual_tokens = self.client.count_tokens(truncated)
        
        return truncated
    
    def run_test(self, file_path: str, max_context: Optional[int] = None,
                 output_file: Optional[str] = None) -> List[Dict]:
        """ Run the complete degradation test
        
        Args:
            file_path: Path to reference text file
            max_context: Maximum context length to test
            output_file: Optional CSV output file
            
        Returns:
            List of test results
        """
        print("=" * 60)
        print("MODEL READABILITY DEGRADATION TEST")
        print("=" * 60)
        
        # Load reference text
        reference_text = self.load_reference_text(file_path)
        
        # Analyze baseline readability
        baseline_analysis = analyze_text_readability(reference_text[:5000])  # First 5k chars
        print(f"\nBaseline text readability:")
        print(f"  Cloze Score: {baseline_analysis['cloze_score']}")
        print(f"  Reading Level: {baseline_analysis['reading_level']}")
        print(f"  Vocabulary Diversity: {baseline_analysis['vocabulary_diversity']}")
        
        # Generate test context lengths
        context_lengths = self.generate_context_lengths(reference_text, max_context)
        
        print(f"\n{'='*60}")
        print("RUNNING DEGRADATION TESTS")
        print(f"{'='*60}")
        
        self.results = []
        
        for i, context_length in enumerate(context_lengths, 1):
            print(f"\n[TEST {i}/{len(context_lengths)}] Context Length: {context_length:,} tokens")
            print("-" * 50)
            
            # Prepare context
            context = self.truncate_to_tokens(reference_text, context_length)
            actual_tokens = self.client.count_tokens(context)
            
            print(f"Actual context tokens: {actual_tokens:,}")
            print(f"Generating continuation...")
            
            # Generate continuation
            continuation = self.client.generate_continuation(context, max_tokens=1024)
            
            if not continuation:
                print(f"WARNING: No continuation generated for context length {context_length}")
                continue
            
            # Analyze readability
            analysis = analyze_text_readability(continuation)
            
            # Store results
            result = {
                'context_length': context_length,
                'actual_context_tokens': actual_tokens,
                'continuation_length': len(continuation),
                'timestamp': datetime.now().isoformat(),
                **analysis
            }
            self.results.append(result)
            
            # Display results
            print(f"\nReadability Analysis:")
            print(f"  Cloze Score: {analysis['cloze_score']:6.2f}")
            print(f"  Reading Level: {analysis['reading_level']:>6}")
            print(f"  Unfamiliar Words: {analysis['pct_unfamiliar_words']*100:5.1f}%")
            print(f"  Avg Sentence Length: {analysis['avg_sentence_length']:5.1f}")
            print(f"  Sentence Variance: {analysis['sentence_length_variance']:5.1f}")
            print(f"  Vocabulary Diversity: {analysis['vocabulary_diversity']:5.3f}")
        
        # Analysis and summary
        self._print_summary()
        
        # Save results
        if output_file:
            self._save_results(output_file)
        
        return self.results
    
    def _print_summary(self):
        """ Print test summary and degradation analysis """
        if len(self.results) < 2:
            return
        
        print(f"\n{'='*60}")
        print("DEGRADATION ANALYSIS")
        print(f"{'='*60}")
        
        # Calculate trends
        cloze_scores = [r['cloze_score'] for r in self.results]
        vocab_diversity = [r['vocabulary_diversity'] for r in self.results]
        sentence_variance = [r['sentence_length_variance'] for r in self.results]
        
        print(f"\nCloze Score Trend:")
        print(f"  Starting: {cloze_scores[0]:6.2f}")
        print(f"  Ending:   {cloze_scores[-1]:6.2f}")
        print(f"  Change:   {cloze_scores[-1] - cloze_scores[0]:+6.2f}")
        
        print(f"\nVocabulary Diversity Trend:")
        print(f"  Starting: {vocab_diversity[0]:6.3f}")
        print(f"  Ending:   {vocab_diversity[-1]:6.3f}")
        print(f"  Change:   {vocab_diversity[-1] - vocab_diversity[0]:+6.3f}")
        
        # Look for degradation points
        degradation_points = []
        
        for i in range(1, len(self.results)):
            prev_score = cloze_scores[i-1]
            curr_score = cloze_scores[i]
            
            # Significant drop in cloze score
            if prev_score - curr_score > 3.0:
                degradation_points.append({
                    'context_length': self.results[i]['context_length'],
                    'metric': 'cloze_score',
                    'drop': prev_score - curr_score
                })
        
        if degradation_points:
            print(f"\n⚠️  DEGRADATION DETECTED:")
            for point in degradation_points:
                print(f"  At {point['context_length']:,} tokens: {point['metric']} dropped {point['drop']:.2f}")
        else:
            print(f"\n✅ No significant degradation detected in tested range")
        
        # Print full results table
        print(f"\nFULL RESULTS:")
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
        
        fieldnames = self.results[0].keys()
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)
        
        print(f"\n📊 Results saved to: {output_file}")


# ================================
# COMMAND LINE INTERFACE
# ================================

def main():
    parser = argparse.ArgumentParser(
        description="Test LLM readability degradation across context lengths",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python readability_test.py novel.txt --api-url http://localhost:5001
  python readability_test.py document.pdf --max-context 16384 --output results.csv
  python readability_test.py text.txt --word-list dale_chall_words.txt
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
        '--output',
        default=None,
        help='Output CSV file for detailed results'
    )
    
    args = parser.parse_args()
    
    try:
        tester = ReadabilityDegradationTester(
            api_url=args.api_url,
            api_password=args.api_password,
            word_list_path=args.word_list
        )
        
        tester.run_test(
            file_path=args.input_file,
            max_context=args.max_context,
            output_file=args.output
        )
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
