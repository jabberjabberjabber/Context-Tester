#!/usr/bin/env python3
"""
API client for text generation using OpenAI-compatible endpoints.

Supports KoboldCpp, NVIDIA NIM, and generic OpenAI-compatible APIs.
"""

from typing import Optional, List
import json
import requests
from find_last_sentence import find_last_sentence_ending
from chunker_regex import chunk_regex
from tokenizer_utils import UnifiedTokenizer


class StreamingAPIClient:
    """Client for generating text continuations via streaming API."""

    def __init__(
        self,
        api_url: str,
        api_password: Optional[str] = None,
        tokenizer_model: Optional[str] = None,
        model_name: Optional[str] = None,
        max_context: Optional[int] = None,
        embedding_model: Optional[str] = None,
        hf_token: Optional[str] = None
    ):
        """
        Initialize API client.

        Args:
            api_url: Base URL for the API
            api_password: Optional API key/bearer token
            tokenizer_model: HuggingFace tokenizer name
            model_name: Model name
            max_context: Maximum context length
            embedding_model: Embedding model name for NVIDIA NIM
            hf_token: HuggingFace token for gated repos
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
        self.embedding_model = embedding_model or "nvidia/nv-embed-v1"
        self._max_context = max_context

        # Initialize unified tokenizer
        base_url = self.api_url.replace('/v1/chat/completions', '')
        self.tokenizer = UnifiedTokenizer(
            api_url=base_url,
            api_headers=self.headers,
            tokenizer_model=tokenizer_model,
            model_name=model_name,
            hf_token=hf_token
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return self.tokenizer.count_tokens(text)

    def tokenize_text_batched(self, text: str) -> List[int]:
        """Tokenize text to list of token IDs."""
        return self.tokenizer.tokenize(text)

    def tokens_to_text_batched(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text."""
        return self.tokenizer.detokenize(token_ids)

    def prune_text(self, text: str, max_context: int = 32768, total_tokens: int = 64536) -> str:
        """
        Get max amount of text that fits into a natural breakpoint.

        Uses binary search over chunk boundaries to find the largest amount
        of text that fits within max_context tokens.
        """
        if total_tokens < max_context:
            return text

        print(f"Pruning text from {total_tokens:,} tokens to fit {max_context:,} tokens at natural boundaries...")

        # Get all chunk boundaries
        matches = list(chunk_regex.finditer(text))
        if not matches:
            print("Warning: No regex matches found, using character-based truncation")
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

    def get_embeddings(self, texts: List[str], model: Optional[str] = None) -> Optional[List[List[float]]]:
        """
        Get embeddings for a list of texts using OpenAI-compatible API.

        Args:
            texts: List of strings to embed
            model: Embedding model name (defaults to self.embedding_model)

        Returns:
            List of embedding vectors, or None if API call fails
        """
        if not texts:
            return None

        model = model or self.embedding_model

        try:
            base_url = self.api_url.replace('/v1/chat/completions', '')
            embeddings_url = f"{base_url}/v1/embeddings"

            payload = {
                "input": texts,
                "model": model,
                "encoding_format": "float",
                "truncate": "NONE"
            }

            response = self._make_request_with_retry(embeddings_url, payload=payload)

            if response.status_code == 200:
                data = response.json()
                if "data" in data:
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

    def get_max_context_length(self) -> int:
        """Get model's maximum context length from API or configuration."""
        # Use configured value if available
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
        """Get model name from API."""
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
        """Make request with automatic retry on rate limits."""
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
                    print(f"Request failed: Too many tokens")
                    return None
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

    def generate_continuation(
        self,
        context: str,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_k: int = 100,
        top_p: float = 1.0
    ) -> str:
        """Generate text continuation from context."""
        instruction = """Continue this story for as long as you can. Do not try to add a conclusion or ending, just keep writing as if this were part of the middle of a novel. Maintain the same style, tone, and narrative voice. Focus on developing the plot, characters, and setting naturally."""

        if not context:
            return None

        context = find_last_sentence_ending(context)
        print(f"Starting from: {context[-100:]}")

        payload = {
            "messages": [
                {"role": "user", "content": f"{context}\n\n{instruction}"}
            ],
            "model": self.model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
        }

        result = []

        try:
            response = self._make_request_with_retry(
                self.api_url,
                payload=payload,
                stream=True
            )

            if not response:
                return ""

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
                    content = data['choices'][0]['delta'].get('content')
                    if content is not None:
                        result.append(content)
                        print(content, end='', flush=True)
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

            print()
            return ''.join(result)

        except Exception as e:
            print(f"\nError in generation: {str(e)}")
            return ""
