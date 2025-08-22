import re

def find_last_sentence_ending(text):
    """
    Finds the last sentence ending point in a string and returns the text up to that point.
    
    Handles various sentence endings including:
    - Standard punctuation (. ! ?)
    - Ellipses (... or more dots)
    - Dialog with quotes
    - Multiple punctuation marks
    
    Args:
        text (str): The input string
        
    Returns:
        str: The string cut at the last sentence ending point, or the original string if no ending found
    """
    if not text or not text.strip():
        return text
    
    # Remove trailing whitespace for processing
    text = text.rstrip()
    
    if not text:
        return text
    
    # Define sentence ending patterns
    # Look for various sentence endings working backwards from the end
    patterns = [
        # Dialog patterns: "text." or 'text!' etc.
        r'["\'][.!?]+["\']',
        # Ellipses (3 or more dots, possibly followed by other punctuation)
        r'\.{3,}[!?]*',
        # Multiple punctuation combinations
        r'[.!?]{2,}',
        # Standard single punctuation
        r'[.!?]'
    ]
    
    # Combine all patterns
    combined_pattern = '|'.join(f'({pattern})' for pattern in patterns)
    
    # Find all matches
    matches = list(re.finditer(combined_pattern, text))
    
    if not matches:
        return text
    
    # Work backwards through matches to find the last valid sentence ending
    for match in reversed(matches):
        end_pos = match.end()
        
        # Get the character before the match (if it exists)
        start_pos = match.start()
        
        # Check if this might be an abbreviation or decimal number
        if match.group().startswith('.') and len(match.group()) == 1:
            # Single period - check for common abbreviation patterns
            if start_pos > 0:
                # Look at context before the period
                before_match = text[:start_pos]
                
                # Check for abbreviations (capital letter or common abbreviations)
                if re.search(r'\b[A-Z]$', before_match):
                    continue  # Likely abbreviation, skip
                
                # Check for decimal numbers
                if re.search(r'\d$', before_match):
                    continue  # Likely decimal number, skip
        
        # This appears to be a valid sentence ending
        return text[:end_pos]
    
    # If no valid sentence ending found, return original text
    return text


def find_last_sentence_ending_simple(text):
    """
    Simplified version that finds the last occurrence of sentence-ending punctuation.
    
    Args:
        text (str): The input string
        
    Returns:
        str: The string cut at the last sentence ending point
    """
    if not text:
        return text
    
    text = text.rstrip()
    
    # Find the last occurrence of sentence-ending punctuation
    last_pos = -1
    
    for i in range(len(text) - 1, -1, -1):
        char = text[i]
        if char in '.!?':
            # Check if it's part of ellipses
            if char == '.' and i >= 2 and text[i-1:i+1] == '..':
                # Find the start of ellipses
                j = i
                while j >= 0 and text[j] == '.':
                    j -= 1
                last_pos = i + 1
                break
            elif char in '!?':
                last_pos = i + 1
                break
            elif char == '.':
                # Single period - basic check for abbreviations
                if i > 0 and text[i-1].isupper() and (i == 1 or text[i-2] == ' '):
                    continue  # Likely abbreviation
                last_pos = i + 1
                break
    
    if last_pos > 0:
        return text[:last_pos]
    else:
        return text


# Example usage and tests
if __name__ == "__main__":
    test_cases = [
        "This is a sentence. This is incomplete",
        "Hello world! How are you? I'm doing",
        'He said, "I am fine." But then he left without',
        "Wait... what happened? Something seems",
        "The value is 3.14159 and that's important. But we need",
        "Dr. Smith went to the store. However, he forgot",
        "Really?! I can't believe it! This is",
        "One sentence. Two sentences. Three sentences.",
        "No ending punctuation here",
        "",
        "   ",
        "Just one sentence.",
        "Multiple...dots...everywhere...but incomplete",
        "Normal text... and then some more",
    ]
    
    print("Testing find_last_sentence_ending function:")
    print("=" * 50)
    
    for i, test in enumerate(test_cases, 1):
        result = find_last_sentence_ending(test)
        print(f"Test {i}:")
        print(f"Input:  '{test}'")
        print(f"Output: '{result}'")
        print()
