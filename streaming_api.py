from typing import Dict, List, Optional, Set, Any, Literal, TypeAlias
import json
import requests
from find_last_sentence import find_last_sentence_ending
from chunker_regex import chunk_regex

# ================================
# API CLIENT FOR TEXT GENERATION
# ================================

class StreamingAPIClient:
    """ Client for generating text continuations via streaming API 
    
    Supports both KoboldCpp (with API tokenization) and NVIDIA NIM
    (with local HuggingFace tokenizer).
    """
    
    def __init__(self, api_url: str, api_password: Optional[str] = None,
                 tokenizer_model: Optional[str] = None, model_name: Optional[str] = None,
                 max_context: Optional[int] = None, embedding_model: Optional[str] = None):
        """ Initialize API client
        
        Args:
            api_url: Base URL for the API
            api_password: Optional API key/bearer token
            tokenizer_model: HuggingFace tokenizer name (for NVIDIA NIM)
            model_name: Model name (will try as tokenizer if tokenizer_model not provided)
            max_context: Maximum context length (required for NVIDIA NIM)
        """
        self.api_url = api_url
        if not self.api_url.endswith('/v1/chat/completions'):
            self.api_url = f"{self.api_url.rstrip('/')}/v1/chat/completions"
            
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        
        if api_password:
            self.headers["Authorization"] = f"Bearer {api_password}"
        self.model_name = model_name
        # Setup local tokenizer and specify embedding model for NVIDIA NIM (or other APIs without tokenization endpoints)
        tokenizer_to_load = tokenizer_model or model_name
        
        if embedding_model:
            self.embedding_model = embedding_model
            
        if tokenizer_to_load:
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_to_load, trust_remote_code=True)
                self._max_context = max_context
                print(f"Loaded local tokenizer: {tokenizer_to_load}")
                if max_context:
                    print(f"Using configured max context: {max_context:,} tokens")
            except Exception as e:
                print(f"Failed to load tokenizer '{tokenizer_to_load}': {e}")
                print("Falling back to API tokenization (KoboldCpp mode)")
                self.tokenizer = None
                self._max_context = None
        else:
            self.tokenizer = None
            self._max_context = None
    
    def get_embeddings(self, texts: List[str], model: str = "nvidia/nv-embed-v1") -> Optional[List[List[float]]]:
        """ Get embeddings for a list of texts using OpenAI-compatible API
        
        Args:
            texts: List of strings to embed
            model: Embedding model name (default for NVIDIA NIM)
        
        Returns:
            List of embedding vectors, or None if API call fails
        """
        if not texts:
            return None
            
        if self.embedding_model:
            model = self.embedding_model
        
        try:
            # Use same base URL but with embeddings endpoint
            base_url = self.api_url.replace('/v1/chat/completions', '')
            embeddings_url = f"{base_url}/v1/embeddings"
            
            payload = {
                "input": texts,
                "model": model,
                "encoding_format": "float",
                "truncate": "NONE"
            }
            
            response = self._make_request_with_retry(
                embeddings_url,
                payload=payload,
                #headers={"Content-Type": "application/json", **{k: v for k, v in self.headers.items() if k == "Authorization"}},
                
            )
            
            if response.status_code == 200:
                data = response.json()
                if "data" in data:
                    # Sort by index to maintain order
                    embeddings = sorted(data["data"], key=lambda x: x.get("index", 0))
                    return [item["embedding"] for item in embeddings]
                else:
                    print("No embedding data in API response")
                    return None
            else:
                print(f"Embeddings API error {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            return None
            
    def count_tokens(self, text: str) -> int:
        """ Count tokens in text
        
        Uses local tokenizer if available, otherwise falls back to API.
        """
        
        tokens = self.tokenize_text_batched(text)
        if tokens:
            return len(tokens)
        else:
            return None
        
            
    def prune_text(self, text: str, max_context: int = 32768, total_tokens: int = 64536):
        """ Get max amount of text that fits into a natural breakpoint
        
        Uses binary search over chunk boundaries to find the largest amount
        of text that fits within max_context tokens when chunked naturally.
        """
        
        if total_tokens >= max_context:
            pass
        else:
            return None
            
        print(f"Pruning text from {total_tokens:,} tokens to fit {max_context:,} tokens at natural boundaries...")
        
        # Get all chunk boundaries
        matches = list(chunk_regex.finditer(text))
        if not matches:
            print("Warning: No regex matches found, using character-based truncation")
            # Fallback: truncate at max_context * 4 characters (rough estimate)
            return text[:max_context * 2]
        
        print(f"Found {len(matches)} natural chunks to work with")
        
        # Binary search over chunk boundaries
        left, right = 0, len(matches)
        best_text = ""
        
        while left < right:
            mid = (left + right + 1) // 2
            
            # Build text up to chunk 'mid'
            end_pos = matches[mid - 1].end()
            candidate_text = text[:end_pos]
            
            # Count tokens for this candidate
            candidate_tokens = self.count_tokens(candidate_text)
            
            if candidate_tokens <= max_context:
                best_text = candidate_text
                left = mid
                print(f"  Chunk {mid:,}/{len(matches):,}: {candidate_tokens:,} tokens - fits")
            else:
                right = mid - 1
                print(f"  Chunk {mid:,}/{len(matches):,}: {candidate_tokens:,} tokens - too large")
        
        final_tokens = self.count_tokens(best_text)
        print(f"Pruned to {final_tokens:,} tokens ({final_tokens/max_context:.1%} of max context)")
        
        return best_text
        
    def tokenize_text_batched(self, text: str, chunk_size: int = 45000) -> List[int]:
        """ Tokenize large text by batching API calls, return token IDs 
        
        Uses local tokenizer if available (no batching needed).
        """
        if not text or not text.strip():
            return []
        
        if self.tokenizer:
            # Local tokenization - fast, no batching needed
            token_ids = self.tokenizer.encode(text, add_special_tokens=True)
            #print(f"Tokenized {len(token_ids):,} tokens locally")
            return token_ids
        
        # KoboldCpp API fallback with batching
        base_url = self.api_url.replace('/v1/chat/completions', '')
        all_token_ids = []
        
        # Split text into manageable chunks
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        #print(f"Tokenizing text in {len(chunks)} chunks...")
        
        for i, chunk in enumerate(chunks, 1):
            try:
                response = requests.post(
                    f"{base_url}/api/extra/tokencount",
                    json={"prompt": chunk},
                    headers={"Content-Type": "application/json"},
                    timeout=9999
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if "ids" in data:
                        chunk_tokens = data["ids"]
                        all_token_ids.extend(chunk_tokens)
                        #print(f"  Chunk {i}/{len(chunks)}: {len(chunk_tokens)} tokens")
                        
                    else:
                        #print(f"  Chunk {i}/{len(chunks)}: No token IDs returned")
                        # Fallback estimation
                        estimated_tokens = int(len(chunk.split()) * 1.33)
                        all_token_ids.extend(range(len(all_token_ids), len(all_token_ids) + estimated_tokens))
                else:
                    print(f"  Chunk {i}/{len(chunks)}: API error {response.status_code}")
                    # Fallback estimation
                    #estimated_tokens = int(len(chunk.split()) * 1.33)
                    #all_token_ids.extend(range(len(all_token_ids), len(all_token_ids) + estimated_tokens))
                    return None
                    
                    
            except Exception as e:
                print(f"  Chunk {i}/{len(chunks)}: Error {e}")
                # Fallback estimation
                #estimated_tokens = int(len(chunk.split()) * 1.33)
                #all_token_ids.extend(range(len(all_token_ids), len(all_token_ids) + estimated_tokens))
                return None
                
        #print(f"Total tokens collected: {len(all_token_ids):,}")
        return all_token_ids
    
    def tokens_to_text_batched(self, token_ids: List[int], chunk_size: int = 45000) -> str:
        """ Detokenize large token array by batching API calls, return text 
        
        Uses local tokenizer if available (no batching needed).
        """
        if not token_ids:
            return ""
        
        if self.tokenizer:
            # Local detokenization - fast, no batching needed
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            #print(f"Detokenized {len(token_ids):,} tokens locally")
            return text
        
        # KoboldCpp API fallback with batching
        base_url = self.api_url.replace('/v1/chat/completions', '')
        all_text = []
        
        # Split text into manageable chunks
        chunks = [token_ids[i:i+chunk_size] for i in range(0, len(token_ids), chunk_size)]
        
        #print(f"Detokenizing text in {len(chunks)} chunks...")
        
        for i, chunk in enumerate(chunks, 1):
            try:
                response = requests.post(
                    f"{base_url}/api/extra/detokenize",
                    json={"ids": chunk},
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if "result" in data:
                        chunk_text = data["result"]
                        all_text.extend(chunk_text)
                        #print(f"  Chunk {i}/{len(chunks)}: {len(chunk_text)} words")
                    else:
                        print(f"Token detokenize failed: success=False")
                        return None
                else:
                    print(f"Token detokenize failed: {response.status_code}")
                    return None
                
            except Exception as e:
                print(f"  Chunk {i}/{len(chunks)}: Error {e}")
                return None
        
        text = "".join(all_text)
        return text
        
    def tokens_to_text(self, token_ids: List[int]) -> str:
        """ Convert token IDs back to text via API 
        
        Uses local tokenizer if available.
        """
        return self.tokenize_text_batched(token_ids)
        
        if not token_ids:
            print(f"No token ids!")
            return ""
        
        if self.tokenizer:
            # Local detokenization
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        
        # KoboldCpp API fallback
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
                print(data)
                if data.get("success", False):
                    return data.get("result", "")
                else:
                    print(f"Token detokenize failed: success=False")
                    return None
            else:
                print(f"Token detokenize failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Token detokenize error: {e}")
            return None
    
    def get_max_context_length(self) -> int:
        """ Get model's maximum context length from API or configuration """
        
        # Use configured value if available (NVIDIA NIM mode)
        if self._max_context:
            return self._max_context
        
        # Try to detect from API (KoboldCpp mode)
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

    def _make_request_with_retry(self, url: str, payload: dict, stream: bool = False, max_retries: int = 5):
        """ Make request with automatic retry on rate limits """
        import time
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    headers=self.headers,
                    stream=stream,
                    timeout=999
                )
                
                if response.status_code == 200:
                    return response
                elif response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    print(f"Rate limited. Waiting {retry_after}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(retry_after)
                elif response.status_code == 400:
                    print(f"Requested too many tokens. Marking as complete")
                    return "[TOO_MANY_TOKENS]"
                else:
                    print(f"Request failed with status {response.status_code}: {response.text}")
                    return None
                    
            except Exception as e:
                print(f"Request error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return None
        
        print(f"Max retries exceeded")
        return None
    
    def generate_continuation(self, context: str, max_tokens: int = 1024,
                            temperature: float = 1.0, top_k: int = 100, 
                            top_p: float = 1.0, min_p: float = 0.0, 
                            rep_pen: float = 1.0) -> str:
        """ Generate text continuation from context """
        
        instruction = """Continue this story for as long as you can. Do not try to add a conclusion or ending, just keep writing as if this were part of the middle of a novel. Maintain the same style, tone, and narrative voice. Focus on developing the plot, characters, and setting naturally."""
        if not context:
            return None
            
        context = find_last_sentence_ending(context)
        print(f"Starting from: {context[-100:]}")
        
        payload = {
            "messages": [
                #{"role": "system", "content": "You are a skilled novelist continuing a story."},
                {"role": "user", "content": f"{context}\n\n{instruction}"}
            ],
            "model": self.model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
            #"top_k": top_k
        }
        
        result = []
        
        try:
            response = self._make_request_with_retry(
                self.api_url,
                payload=payload,
                stream=True
            )
            
            if response == "[TOO_MANY_TOKENS]":
                print()
                return ""
                
            for line in response.iter_lines():
                if not line:
                    continue
                    
                line_text = line.decode('utf-8')
                
                if line_text.startswith('data: '):
                    line_text = line_text[6:]  # Remove 'data: ' prefix
                
                if line_text == '[DONE]':
                    break
                    
                try:
                    data = json.loads(line_text)
                    content = data['choices'][0]['delta'].get('content')
                    if content is not None:  # Explicitly check for None, allow empty strings
                        result.append(content)
                        print(content, end='', flush=True)
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
            
            print()  # New line after generation
            return ''.join(result)
                
        except Exception as e:
            print(f"\nError in generation: {str(e)}")
            return ""
