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
import os

import requests
from readability_tests import (
    initialize_word_list, 
    analyze_text_comprehensive, 
    analyze_text_readability,
    reading_level_from_cloze,
    is_power_of_two
)

from generate_plot import make_png
from chunker_regex import chunk_regex
from find_last_sentence import find_last_sentence_ending

from requests.exceptions import RequestException
from bs4 import BeautifulSoup
from extractous import Extractor
from streaming_api import StreamingAPIClient

from outlier_detection import (
    analyze_round_outliers, 
    filter_outliers_from_results,
    print_outlier_summary
)

warnings.filterwarnings("ignore", category=UserWarning, module="bs4")

# ================================
# MAIN DEGRADATION TESTER
# ================================

class ReadabilityDegradationTester:
    """ Tests model degradation across increasing context lengths with fixed continuation point """
    
    def __init__(self, api_url: str, api_password: Optional[str] = None,
                 word_list_path: str = "easy_words.txt", num_rounds: int = 1, divisions: int = 1,
                 model_name: Optional[str] = None, max_tokens: int = 1024, 
                 temperature: float = 1.0, top_k: int = 100, top_p: float = 1.0, 
                 min_p: float = 0.1, rep_pen: float = 1.01, start_context: Optional[int] = None,
                 text_name: str = "unknown"):
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
            start_context: Optional starting context size (skip smaller sizes)
            text_name: Name of the source text file
        """
        self.client = StreamingAPIClient(api_url, api_password)
        initialize_word_list(word_list_path)
        self.num_rounds = max(1, num_rounds)
        self.results = []
        self.detailed_results = []  # Store individual round results
        self.comprehensive_data = {}  # Store ALL data for JSON export
        self.divisions = max(1, divisions)
        self.all_tokens = []  # Full tokenized text
        self.working_tokens = []  # Last max_context tokens
        self.start_context = start_context  # Optional starting context size
        self.text_name = text_name  # Fix: store text_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p
        self.rep_pen = rep_pen
        
        self.model_name = model_name or self.client.get_model_name() or "unknown-model"
    
    def generate_output_filename(self, extension: str = "csv") -> str:
        """ Generate output filename based on model name and test parameters """
        clean_model = self.model_name.replace('/', '-').replace('\\', '-').replace(':', '-')
        clean_text_name = self.text_name.replace('/', '-').replace('\\', '-').replace(':', '-')
        filename = f"{clean_model}-{clean_text_name}-{self.num_rounds}r{self.divisions}d.{extension}"
        return filename
        
    def normalize_content(self, content: str) -> str:
        """ Convert fixed-width text and normalize characters """
        content = unicodedata.normalize('NFKC', content)
        content = content.replace('--', '—')             
        text = content.replace('\r\n', '\n').replace('\r', '\n')

        paragraphs = text.split('\n\n')
        
        # Fix windows line breaks and convert fixed width text to wrap
        result = '\n\n'.join(para.replace('\n', ' ') for para in paragraphs)
        result = result.replace('\n\n', '\n\n    ')
        return result
        
    def load_reference_text(self, file_path: str) -> str:
        """ Load reference text from file """
        extractor = Extractor()
        
        content, metadata = extractor.extract_file_to_string(file_path)
        print(f"Loaded reference text: {metadata.get('resourceName', file_path)}")
        print(f"Content type: {metadata.get('Content-Type', 'Unknown')}")
        
        # Debug text loading
        if not content or not content.strip():
            raise ValueError(f"File {file_path} appears to be empty or could not be read")
        content_size = int(len(content) * 0.8)
        normalized_content = self.normalize_content(content[:content_size])
        print(f"Text preview (first 200 chars): {normalized_content[:200]!r}")
        print(f"Total characters: {len(normalized_content):,}")
        return normalized_content
    
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
        
        working_size = max_context 
        pruned_text = self.client.prune_text(text, max_context)
        # Tokenize entire text 
        self.all_tokens = self.client.tokenize_text_batched(pruned_text)
        
        if not self.all_tokens:
            print("ERROR: Failed to tokenize text")
            return False
        
        total_tokens = len(self.all_tokens)
        print(f"Total tokens in reference text: {total_tokens:,}")
        
        # Check if we have enough tokens
        min_required = int(max_context * 1.2)
        if total_tokens < min_required:
            print(f"ERROR: Insufficient text. Need at least {min_required:,} tokens, got {total_tokens:,} after pruning")
            return False
        
        # Take the last max_context tokens as our working set
        self.working_tokens = self.all_tokens[-min_required:]
        print(f"Working with {len(self.working_tokens):,} tokens for testing")
        
        # Store tokenization info for JSON
        self.comprehensive_data['tokenization'] = {
            'original_text_chars': len(text),
            'pruned_text_chars': len(pruned_text),
            'total_tokens': total_tokens,
            'working_tokens_count': len(self.working_tokens),
            'min_required_tokens': min_required,
            'pruned_text_preview': pruned_text[:500] + "..." if len(pruned_text) > 500 else pruned_text
        }
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
        
        # Filter tiers based on start_context if specified
        if self.start_context is not None:
            tiers = [tier for tier in tiers if tier >= self.start_context]
            if not tiers:
                # If start_context is larger than all tiers, just use max_context
                return [min(self.start_context, max_context)]
            print(f"Starting from context size: {self.start_context:,} tokens")
        
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
            Context text for this tier, trimmed to natural boundaries
        """
        if context_length > len(self.working_tokens):
            # Use all available working tokens
            context_tokens = self.working_tokens
            context_text = self.client.tokens_to_text(context_tokens)
            return context_text
        
        # Convert ALL working tokens to text to establish fixed ending position
        full_text = self.client.tokens_to_text(self.working_tokens)
        
        if not full_text:
            return ""
        
        # Find natural chunk boundaries in the full text
        matches = list(chunk_regex.finditer(full_text))
        if not matches:
            # Fallback: just truncate to target length if no regex matches
            target_tokens = self.working_tokens[-context_length:]
            fallback_text = self.client.tokens_to_text(target_tokens)
            print(f"  Context window: {context_length:,} tokens (fallback: no regex matches)")
            return fallback_text
        
        # Find the best starting chunk boundary to get close to target tokens
        best_text = ""
        best_token_diff = float('inf')
        
        # Try each chunk boundary as a potential starting point
        for match in matches:
            start_pos = match.start()
            candidate_text = full_text[start_pos:]  # From this boundary to end
            candidate_tokens = self.client.count_tokens(candidate_text)
            
            # Calculate how close this is to our target
            token_diff = abs(candidate_tokens - context_length)
            
            # Prefer candidates that are close to target, with slight preference for under-target
            if candidate_tokens <= context_length:
                adjusted_diff = token_diff  # No penalty for being under target
            else:
                adjusted_diff = token_diff + 100  # Small penalty for being over target
            
            if adjusted_diff < best_token_diff:
                best_text = candidate_text
                best_token_diff = adjusted_diff
        
        if best_text:
            actual_tokens = self.client.count_tokens(best_text)
            print(f"  Context window: {actual_tokens:,} tokens (target: {context_length:,}) - starts at natural boundary")
            return best_text
        else:
            # Final fallback
            target_tokens = self.working_tokens[-context_length:]
            fallback_text = self.client.tokens_to_text(target_tokens)
            print(f"  Context window: {context_length:,} tokens (fallback: exact token slice)")
            return fallback_text
    
    def average_results(self, round_results: List[Dict]) -> Dict[str, Any]:
        """ Average results across multiple rounds """
        if not round_results:
            return {}
        
        if len(round_results) == 1:
            result = round_results[0].copy()
            result['num_rounds'] = 1
            result['cloze_score_std'] = 0.0
            result['vocab_diversity_std'] = 0.0
            result['timestamp'] = datetime.now().isoformat()
            return result
        
        numerical_fields = [
            'pct_unfamiliar_words', 'avg_sentence_length', 'cloze_score',
            'word_count', 'sentence_count', 'sentence_length_variance',
            'vocabulary_diversity', 'continuation_length',
            'bigram_repetition_rate', 'trigram_repetition_rate', 'unique_word_ratio_100',
            'word_entropy', 'char_entropy',
            'comma_density', 'semicolon_density', 'question_density', 'exclamation_density',
            'avg_syllables_per_word', 'long_word_ratio', 'function_word_ratio',
            'sentence_length_skewness', 'sentence_length_kurtosis', 'avg_word_length'
        ]
        
        averaged = {}
        
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
        
        if 'cloze_score' in averaged:
            averaged['reading_level'] = reading_level_from_cloze(averaged['cloze_score'])
        
        averaged['num_rounds'] = len(round_results)
        
        cloze_values = [r['cloze_score'] for r in round_results if 'cloze_score' in r and r['cloze_score'] is not None]
        vocab_values = [r['vocabulary_diversity'] for r in round_results if 'vocabulary_diversity' in r and r['vocabulary_diversity'] is not None]
        
        averaged['cloze_score_std'] = round(statistics.stdev(cloze_values), 3) if len(cloze_values) > 1 else 0.0
        averaged['vocab_diversity_std'] = round(statistics.stdev(vocab_values), 4) if len(vocab_values) > 1 else 0.0
        
        averaged['timestamp'] = datetime.now().isoformat()
        
        return averaged
    
    def run_test(self, file_path: str, max_context: Optional[int] = None) -> List[Dict]:
        """ Run the complete degradation test
        
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
        
        test_start_time = datetime.now()
        self.comprehensive_data = {
            'experiment_metadata': {
                'model_name': self.model_name,
                'source_text_file': file_path,
                'source_text_name': self.text_name,
                'start_time': test_start_time.isoformat(),
                'num_rounds': self.num_rounds,
                'divisions': self.divisions,
                'start_context': self.start_context,
                'generation_params': {
                    'max_tokens': self.max_tokens,
                    'temperature': self.temperature,
                    'top_k': self.top_k,
                    'top_p': self.top_p,
                    'min_p': self.min_p,
                    'rep_pen': self.rep_pen
                },
                'api_url': self.client.api_url
            },
            'rounds': [],
            'errors': [],
            #'outlier_analysis': {},
            'summary_stats': {}
        }
        
        try:
            reference_text = self.load_reference_text(file_path)
            self.comprehensive_data['reference_text'] = {
                'length_chars': len(reference_text),
                'preview': reference_text[:1000] + "..." if len(reference_text) > 1000 else reference_text
            }
        except Exception as e:
            error_msg = f"Failed to load reference text: {e}"
            print(error_msg)
            self.comprehensive_data['errors'].append({
                'timestamp': datetime.now().isoformat(),
                'type': 'text_loading',
                'message': error_msg
            })
            return []
        
        if max_context is None:
            max_context = self.client.get_max_context_length()
        else:
            print(f"Using user-specified max context: {max_context:,}")
        
        self.comprehensive_data['experiment_metadata']['max_context'] = max_context
        
        if not self.prepare_tokenized_text(reference_text, max_context):
            error_msg = "Failed to prepare tokenized text"
            print(error_msg)
            self.comprehensive_data['errors'].append({
                'timestamp': datetime.now().isoformat(),
                'type': 'tokenization',
                'message': error_msg
            })
            return []
        
        # Analyze baseline readability
        baseline_text = self.client.tokens_to_text(self.working_tokens[-5000:] if len(self.working_tokens) > 5000 else self.working_tokens)
        baseline_analysis = analyze_text_comprehensive(baseline_text)
        print(f"\nBaseline text readability (from continuation point area):")
        print(f"  Cloze Score: {baseline_analysis['cloze_score']}")
        print(f"  Reading Level: {baseline_analysis['reading_level']}")
        print(f"  Vocabulary Diversity: {baseline_analysis['vocabulary_diversity']}")
        
        self.comprehensive_data['baseline_analysis'] = baseline_analysis
        
        # Generate test context lengths
        context_lengths = self.generate_context_lengths(max_context)
        self.comprehensive_data['context_lengths'] = context_lengths
        
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
            
            # Build context window
            try:
                context = self.build_context_window(context_length)
            except Exception as e:
                error_msg = f"Failed to build context for {context_length} tokens: {e}"
                print(f"WARNING: {error_msg}")
                self.comprehensive_data['errors'].append({
                    'timestamp': datetime.now().isoformat(),
                    'type': 'context_building',
                    'context_length': context_length,
                    'message': error_msg
                })
                continue
            
            if not context:
                print(f"WARNING: Failed to build context for {context_length} tokens")
                continue
            
            print(f"Context built successfully ({len(context.split())} words)")
            
            context_info = {
                'context_length_target': context_length,
                'context_length_actual': self.client.count_tokens(context),
                'context_text_chars': len(context),
                'context_word_count': len(context.split()),
                'context_preview': context[-500:] if len(context) > 500 else context  # Last 500 chars to see ending
            }
            
            round_results = []
            
            for round_num in range(self.num_rounds):
                if self.num_rounds > 1:
                    print(f"\n  Round {round_num + 1}/{self.num_rounds}:")
                else:
                    print(f"Generating continuation...")
                
                round_start_time = datetime.now()
                
                try:
                    continuation = self.client.generate_continuation(
                        context, 
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        top_k=self.top_k,
                        top_p=self.top_p,
                        min_p=self.min_p,
                        rep_pen=self.rep_pen
                    )
                except Exception as e:
                    error_msg = f"Generation failed for round {round_num + 1}: {e}"
                    print(f"WARNING: {error_msg}")
                    self.comprehensive_data['errors'].append({
                        'timestamp': datetime.now().isoformat(),
                        'type': 'generation',
                        'context_length': context_length,
                        'round_number': round_num + 1,
                        'message': error_msg
                    })
                    continue
                
                if not continuation:
                    print(f"WARNING: No continuation generated for round {round_num + 1}")
                    continue
                
                round_end_time = datetime.now()
                generation_time = (round_end_time - round_start_time).total_seconds()
                
                try:
                    analysis = analyze_text_comprehensive(continuation)
                except Exception as e:
                    error_msg = f"Analysis failed for round {round_num + 1}: {e}"
                    print(f"WARNING: {error_msg}")
                    self.comprehensive_data['errors'].append({
                        'timestamp': datetime.now().isoformat(),
                        'type': 'analysis',
                        'context_length': context_length,
                        'round_number': round_num + 1,
                        'message': error_msg
                    })
                    continue
                
                detailed_result = {
                    'context_length': context_length,
                    'actual_context_tokens': context_info['context_length_actual'],
                    'round_number': round_num + 1,
                    'continuation_length': len(continuation),
                    'continuation_tokens': self.client.count_tokens(continuation),
                    'generation_time_seconds': generation_time,
                    'timestamp': round_end_time.isoformat(),
                    **analysis
                }
                
                round_results.append(detailed_result)
                self.detailed_results.append(detailed_result)
                
                round_data = {
                    'round_id': f"{context_length}_{round_num + 1}",
                    'context_length': context_length,
                    'round_number': round_num + 1,
                    'timestamps': {
                        'start': round_start_time.isoformat(),
                        'end': round_end_time.isoformat(),
                        'generation_time_seconds': generation_time
                    },
                    'context_info': context_info,
                    'llm_output': continuation,
                    'continuation_stats': {
                        'length_chars': len(continuation),
                        'length_tokens': self.client.count_tokens(continuation),
                        'word_count': len(continuation.split())
                    },
                    'analysis': analysis,
                    'generation_params_used': {
                        'max_tokens': self.max_tokens,
                        'temperature': self.temperature,
                        'top_k': self.top_k,
                        'top_p': self.top_p,
                        'min_p': self.min_p,
                        'rep_pen': self.rep_pen
                    }
                }
                
                self.comprehensive_data['rounds'].append(round_data)
                
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
            # Skip for now
            pass

            try:
                outlier_analysis = analyze_round_outliers(self.detailed_results)
                self.comprehensive_data['outlier_analysis'] = outlier_analysis
                
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
                error_msg = f"Outlier analysis failed: {e}"
                print(f"\n⚠️  {error_msg}")
                print("Continuing with original results...")
                self.comprehensive_data['errors'].append({
                    'timestamp': datetime.now().isoformat(),
                    'type': 'outlier_analysis',
                    'message': error_msg
                })
        
        # Finalize comprehensive data
        test_end_time = datetime.now()
        self.comprehensive_data['experiment_metadata']['end_time'] = test_end_time.isoformat()
        self.comprehensive_data['experiment_metadata']['total_duration_seconds'] = (test_end_time - test_start_time).total_seconds()
        
        # Calculate summary statistics
        if self.results:
            self.comprehensive_data['summary_stats'] = self._calculate_summary_stats()
        
        # Analysis and summary
        self._print_summary()
        
        # Auto-generate filenames and save results
        csv_file = self.generate_output_filename("csv")
        json_file = self.generate_output_filename("json")

        self._save_results(csv_file)
        #self._save_comprehensive_json(json_file)
        
        return self.results
    
    def _calculate_summary_stats(self) -> Dict[str, Any]:
        """ Calculate summary statistics across all results """
        if not self.results:
            return {}
        
        metrics = ['cloze_score', 'vocabulary_diversity', 'sentence_length_variance', 
                  'word_entropy', 'char_entropy', 'bigram_repetition_rate', 'trigram_repetition_rate']
        
        summary = {}
        for metric in metrics:
            values = [r[metric] for r in self.results if metric in r]
            if values:
                summary[f"{metric}_trend"] = {
                    'start': values[0],
                    'end': values[-1],
                    'change': values[-1] - values[0],
                    'min': min(values),
                    'max': max(values),
                    'mean': statistics.mean(values),
                    'stdev': statistics.stdev(values) if len(values) > 1 else 0.0
                }
        
        return summary
    
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
        
        # Print full results table (keeping backward compatibility - only showing main metrics)
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
        """ Save results to CSV (backward compatible) """
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
        try:
            success = make_png(enhanced_results, output_file)
        except Exception as e:
            print("Plot generation failed!")
            
    def _save_comprehensive_json(self, output_file: str):
        """ Save comprehensive data to JSON """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.comprehensive_json, f, indent=2, ensure_ascii=False)
            print(f"📊 Comprehensive data saved to: {output_file}")
            print(f"   Contains: {len(self.comprehensive_json['rounds'])} rounds, "
                  f"{len(self.comprehensive_json.get('errors', []))} errors, "
                  f"full LLM outputs & analysis")
        except Exception as e:
            print(f"⚠️  Failed to save JSON: {e}")


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
  python main.py novel.txt --start-context 8192  # Skip testing small contexts
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
    
    parser.add_argument(
        '--start-context',
        type=int,
        default=None,
        help='Starting context size in tokens (skip smaller sizes)'
    )
    args = parser.parse_args()
    
    if not is_power_of_two(args.divisions):
        print(f"Divisions must be 1 or a power of 2 such as 2 or 4 or 8")
        return 1
    
    text_name = os.path.basename(args.input_file)
    
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
            rep_pen=args.rep_pen,
            start_context=args.start_context,
            text_name=text_name
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