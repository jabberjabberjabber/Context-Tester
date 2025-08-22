# ================================
# READABILITY AND DEGRADATION ANALYSIS COMPONENTS
# ================================

import re
import math
import statistics
from collections import Counter, defaultdict
from typing import Any, Dict, List, Set, Tuple, Literal, TypeAlias
from bs4 import BeautifulSoup

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

# Common function words for stylometric analysis
FUNCTION_WORDS = {
    'the', 'and', 'of', 'to', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 
    'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 
    'by', 'from', 'they', 'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 
    'there', 'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 
    'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 
    'him', 'know', 'take', 'people', 'into', 'year', 'your', 'good', 'some', 
    'could', 'them', 'see', 'other', 'than', 'then', 'now', 'look', 'only', 
    'come', 'its', 'over', 'think', 'also', 'back', 'after', 'use', 'two', 
    'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 
    'because', 'any', 'these', 'give', 'day', 'most', 'us'
}

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

def sentence_length_variance(text: str) -> float:
    """ Calculate variance in sentence lengths """
    cleaned_text = text.replace("\n", " ").strip()
    sentences = re.findall(r"\b[^.!?]+[.!?]*", cleaned_text, re.UNICODE)
    
    if len(sentences) < 2:
        return 0.0
        
    lengths = [len(sentence.split()) for sentence in sentences]
    return statistics.variance(lengths) if len(lengths) > 1 else 0.0

def vocabulary_diversity(text: str) -> float:
    """ Calculate type-token ratio (vocabulary diversity) """
    words = _words(text)
    if not words:
        return 0.0
    unique_words = set(words)
    return len(unique_words) / len(words)

# ================================
# NEW DEGRADATION METRICS
# ================================

def calculate_repetition_metrics(text: str) -> Dict[str, float]:
    """ Calculate various repetition metrics to detect model loops """
    words = _words(text)
    if len(words) < 3:
        return {
            'bigram_repetition_rate': 0.0,
            'trigram_repetition_rate': 0.0,
            'unique_word_ratio_100': 0.0
        }
    
    # N-gram repetition rates
    bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
    trigrams = [(words[i], words[i+1], words[i+2]) for i in range(len(words)-2)]
    
    bigram_repetition = (len(bigrams) - len(set(bigrams))) / len(bigrams) if bigrams else 0.0
    trigram_repetition = (len(trigrams) - len(set(trigrams))) / len(trigrams) if trigrams else 0.0
    
    # Unique words per 100-token sliding window
    window_size = min(100, len(words))
    if len(words) >= window_size:
        unique_ratios = []
        for i in range(len(words) - window_size + 1):
            window = words[i:i + window_size]
            unique_ratio = len(set(window)) / len(window)
            unique_ratios.append(unique_ratio)
        avg_unique_ratio = statistics.mean(unique_ratios)
    else:
        avg_unique_ratio = len(set(words)) / len(words)
    
    return {
        'bigram_repetition_rate': round(bigram_repetition, 4),
        'trigram_repetition_rate': round(trigram_repetition, 4),
        'unique_word_ratio_100': round(avg_unique_ratio, 4)
    }

def calculate_entropy_metrics(text: str) -> Dict[str, float]:
    """ Calculate entropy measures for text predictability """
    words = _words(text)
    chars = [c for c in text.lower() if c.isalnum() or c.isspace()]
    
    if not words or not chars:
        return {
            'word_entropy': 0.0,
            'char_entropy': 0.0
        }
    
    # Word-level entropy
    word_counts = Counter(words)
    word_total = len(words)
    word_entropy = -sum((count/word_total) * math.log2(count/word_total) 
                       for count in word_counts.values())
    
    # Character-level entropy
    char_counts = Counter(chars)
    char_total = len(chars)
    char_entropy = -sum((count/char_total) * math.log2(count/char_total) 
                       for count in char_counts.values())
    
    return {
        'word_entropy': round(word_entropy, 4),
        'char_entropy': round(char_entropy, 4)
    }

def calculate_stylometric_metrics(text: str) -> Dict[str, float]:
    """ Calculate stylometric consistency metrics """
    words = _words(text)
    cleaned_text = text.replace("\n", " ").strip()
    sentences = re.findall(r"\b[^.!?]+[.!?]*", cleaned_text, re.UNICODE)
    
    if not words or not sentences:
        return {
            'comma_density': 0.0,
            'semicolon_density': 0.0,
            'question_density': 0.0,
            'exclamation_density': 0.0,
            'avg_syllables_per_word': 0.0,
            'long_word_ratio': 0.0,
            'function_word_ratio': 0.0
        }
    
    # Punctuation densities
    comma_count = text.count(',')
    semicolon_count = text.count(';')
    question_count = text.count('?')
    exclamation_count = text.count('!')
    sentence_count = len(sentences)
    
    # Word sophistication metrics
    total_syllables = sum(_count_syllables(word) for word in words)
    long_words = sum(1 for word in words if len(word) > 6)
    function_words = sum(1 for word in words if word in FUNCTION_WORDS)
    
    return {
        'comma_density': round(comma_count / sentence_count, 4),
        'semicolon_density': round(semicolon_count / sentence_count, 4),
        'question_density': round(question_count / sentence_count, 4),
        'exclamation_density': round(exclamation_count / sentence_count, 4),
        'avg_syllables_per_word': round(total_syllables / len(words), 4),
        'long_word_ratio': round(long_words / len(words), 4),
        'function_word_ratio': round(function_words / len(words), 4)
    }

def calculate_structural_metrics(text: str) -> Dict[str, float]:
    """ Calculate structural stability metrics """
    cleaned_text = text.replace("\n", " ").strip()
    sentences = re.findall(r"\b[^.!?]+[.!?]*", cleaned_text, re.UNICODE)
    words = _words(text)
    
    if len(sentences) < 3 or not words:
        return {
            'sentence_length_skewness': 0.0,
            'sentence_length_kurtosis': 0.0,
            'avg_word_length': 0.0
        }
    
    # Sentence length distribution shape
    lengths = [len(sentence.split()) for sentence in sentences]
    
    # Calculate skewness manually (scipy-free)
    mean_length = statistics.mean(lengths)
    variance = statistics.variance(lengths)
    std_dev = math.sqrt(variance)
    
    if std_dev == 0:
        skewness = 0.0
        kurtosis = 0.0
    else:
        # Skewness: third moment / std_dev^3
        third_moment = sum((x - mean_length)**3 for x in lengths) / len(lengths)
        skewness = third_moment / (std_dev**3)
        
        # Kurtosis: fourth moment / std_dev^4 - 3 (excess kurtosis)
        fourth_moment = sum((x - mean_length)**4 for x in lengths) / len(lengths)
        kurtosis = (fourth_moment / (std_dev**4)) - 3
    
    # Average word length
    total_chars = sum(len(word) for word in words)
    avg_word_length = total_chars / len(words)
    
    return {
        'sentence_length_skewness': round(skewness, 4),
        'sentence_length_kurtosis': round(kurtosis, 4),
        'avg_word_length': round(avg_word_length, 4)
    }

# ================================
# UTILITY FUNCTIONS
# ================================

def _words(in_text: str) -> Tuple[str, ...]:
    """ Extract normalized words from text """
    plain_text = BeautifulSoup(in_text, "html.parser").text
    return tuple(w.lower().strip('.(),"\'') for w in plain_text.split() if w.strip())

def _is_unfamiliar(word: str) -> bool:
    """ Check if word is unfamiliar (not in Dale-Chall list) """
    if word.isdigit():
        return False
    return word not in EASY_WORDS

def _count_syllables(word: str) -> int:
    """ Estimate syllable count for a word """
    word = word.lower()
    if len(word) <= 3:
        return 1
    
    # Count vowel groups
    vowels = 'aeiouy'
    syllable_count = 0
    prev_was_vowel = False
    
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_was_vowel:
            syllable_count += 1
        prev_was_vowel = is_vowel
    
    # Handle silent e
    if word.endswith('e'):
        syllable_count -= 1
    
    # Ensure at least 1 syllable
    return max(1, syllable_count)

def is_power_of_two(n: int) -> bool:
    """Check if n is a power of 2"""
    return n > 0 and (n & (n - 1)) == 0

# ================================
# MAIN ANALYSIS FUNCTION
# ================================

def analyze_text_comprehensive(text: str) -> Dict[str, Any]:
    """ Complete readability and degradation analysis """
    if not text.strip():
        return {
            # Original metrics
            'pct_unfamiliar_words': 0.0,
            'avg_sentence_length': 0.0,
            'cloze_score': 64.0,
            'reading_level': "1",
            'word_count': 0,
            'sentence_count': 0,
            'sentence_length_variance': 0.0,
            'vocabulary_diversity': 0.0,
            # New repetition metrics
            'bigram_repetition_rate': 0.0,
            'trigram_repetition_rate': 0.0,
            'unique_word_ratio_100': 0.0,
            # New entropy metrics
            'word_entropy': 0.0,
            'char_entropy': 0.0,
            # New stylometric metrics
            'comma_density': 0.0,
            'semicolon_density': 0.0,
            'question_density': 0.0,
            'exclamation_density': 0.0,
            'avg_syllables_per_word': 0.0,
            'long_word_ratio': 0.0,
            'function_word_ratio': 0.0,
            # New structural metrics
            'sentence_length_skewness': 0.0,
            'sentence_length_kurtosis': 0.0,
            'avg_word_length': 0.0
        }
    
    # Calculate original metrics
    words = _words(text)
    pct_unfamiliar = pct_unfamiliar_words(text)
    avg_sent_len = avg_sentence_length(text)
    cloze_score = compute_cloze_score(pct_unfamiliar, avg_sent_len)
    reading_level = reading_level_from_cloze(cloze_score)
    vocab_diversity = vocabulary_diversity(text)
    sent_variance = sentence_length_variance(text)
    
    cleaned_text = text.replace("\n", " ").strip()
    sentences = re.findall(r"\b[^.!?]+[.!?]*", cleaned_text, re.UNICODE)
    
    # Calculate new metrics
    repetition_metrics = calculate_repetition_metrics(text)
    entropy_metrics = calculate_entropy_metrics(text)
    stylometric_metrics = calculate_stylometric_metrics(text)
    structural_metrics = calculate_structural_metrics(text)
    
    # Combine all metrics
    result = {
        # Original metrics
        'pct_unfamiliar_words': round(pct_unfamiliar, 4),
        'avg_sentence_length': round(avg_sent_len, 2),
        'cloze_score': cloze_score,
        'reading_level': reading_level,
        'word_count': len(words),
        'sentence_count': len(sentences),
        'sentence_length_variance': round(sent_variance, 2),
        'vocabulary_diversity': round(vocab_diversity, 4)
    }
    
    # Add new metrics
    result.update(repetition_metrics)
    result.update(entropy_metrics)
    result.update(stylometric_metrics)
    result.update(structural_metrics)
    
    return result

# Legacy function for backward compatibility
def analyze_text_readability(text: str) -> Dict[str, Any]:
    """ Legacy function - returns only original metrics for backward compatibility """
    full_analysis = analyze_text_comprehensive(text)
    return {
        'pct_unfamiliar_words': full_analysis['pct_unfamiliar_words'],
        'avg_sentence_length': full_analysis['avg_sentence_length'],
        'cloze_score': full_analysis['cloze_score'],
        'reading_level': full_analysis['reading_level'],
        'word_count': full_analysis['word_count'],
        'sentence_count': full_analysis['sentence_count'],
        'sentence_length_variance': full_analysis['sentence_length_variance'],
        'vocabulary_diversity': full_analysis['vocabulary_diversity']
    }