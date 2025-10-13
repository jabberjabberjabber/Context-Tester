#!/usr/bin/env python3
"""
Unified tokenizer utilities with auto-discovery for multiple backends.

Supports:
- HuggingFace transformers (local tokenization)
- KoboldCpp API (remote tokenization)
- Tiktoken (OpenAI models)
- Automatic fallback chain
"""

import os
import requests
from typing import Optional, List, Union
from pathlib import Path


class UnifiedTokenizer:
    """Unified tokenizer interface supporting multiple backends."""

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_headers: Optional[dict] = None,
        tokenizer_model: Optional[str] = None,
        model_name: Optional[str] = None,
        hf_token: Optional[str] = None
    ):
        """
        Initialize unified tokenizer with automatic backend detection.

        Args:
            api_url: Base API URL (without /v1/chat/completions)
            api_headers: Headers for API requests
            tokenizer_model: HuggingFace model name for tokenizer
            model_name: Model name (used as fallback if tokenizer_model not provided)
            hf_token: HuggingFace token for gated repositories
        """
        self.api_url = api_url
        self.api_headers = api_headers or {}
        self.tokenizer = None
        self.backend = None

        # Try to get HF token from environment if not provided
        if not hf_token:
            hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')

        # Attempt to load tokenizer with fallback chain
        tokenizer_name = tokenizer_model or model_name

        if tokenizer_name:
            self.tokenizer, self.backend = self._load_hf_tokenizer(tokenizer_name, hf_token)

        # Fallback to API tokenization if HF failed
        if not self.tokenizer and not tokenizer_model:
            if self._test_api_tokenization():
                self.backend = "koboldcpp_api"
                print(f"Using KoboldCpp API tokenization")

        # Final fallback to tiktoken
        #if not self.tokenizer and not self.backend:
        #    self.tokenizer, self.backend = self._load_tiktoken(model_name)

        if not self.tokenizer and not self.backend:
            raise RuntimeError(
                "Could not initialize any tokenizer backend. "
                "Please provide a valid --tokenizer-model or ensure API supports tokenization."
            )

    def _load_hf_tokenizer(self, model_name: str, hf_token: Optional[str] = None):
        """Attempt to load HuggingFace tokenizer with auto-discovery."""
        try:
            from transformers import AutoTokenizer

            # Try loading with token (handles gated repos)
            load_kwargs = {'trust_remote_code': True}
            if hf_token:
                load_kwargs['token'] = hf_token

            print(f"Attempting to load HuggingFace tokenizer: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
            print(f"✓ Loaded HuggingFace tokenizer: {model_name}")
            return tokenizer, "huggingface"

        except ImportError:
            print("transformers library not installed, skipping HuggingFace tokenizer")
            return None, None

        except Exception as e:
            error_msg = str(e).lower()

            # Check if it's a gated repo issue
            if 'gated' in error_msg or 'access' in error_msg or '401' in error_msg or '403' in error_msg:
                print(f"Model '{model_name}' requires authorization.")
                print("Please provide HuggingFace token via:")
                print("  1. --hf-token argument")
                print("  2. HF_TOKEN environment variable")
                print(f"  3. Visit https://huggingface.co/{model_name} to accept terms")

            else:
                print(f"Could not load HuggingFace tokenizer '{model_name}': {e}")

            return None, None

    def _test_api_tokenization(self) -> bool:
        """Test if API supports tokenization endpoint."""
        try:
            test_url = f"{self.api_url.rstrip('/')}/api/extra/tokencount"
            response = requests.post(
                test_url,
                json={"prompt": "test"},
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            return response.status_code == 200 and "ids" in response.json()
        except Exception:
            return False

    def _load_tiktoken(self, model_name: Optional[str] = None):
        """Attempt to load tiktoken for OpenAI models."""
        try:
            import tiktoken

            # Map common model names to tiktoken encodings
            encoding_map = {
                'gpt-4': 'cl100k_base',
                'gpt-3.5': 'cl100k_base',
                'gpt-3.5-turbo': 'cl100k_base',
                'text-davinci': 'p50k_base',
            }

            encoding_name = 'cl100k_base'  # Default to GPT-4 encoding
            if model_name:
                for key, value in encoding_map.items():
                    if key in model_name.lower():
                        encoding_name = value
                        break

            tokenizer = tiktoken.get_encoding(encoding_name)
            print(f"✓ Loaded tiktoken encoding: {encoding_name}")
            return tokenizer, "tiktoken"

        except ImportError:
            print("tiktoken library not installed")
            return None, None
        except Exception as e:
            print(f"Could not load tiktoken: {e}")
            return None, None

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if not text or not text.strip():
            return 0

        if self.backend == "huggingface":
            return len(self.tokenizer.encode(text, add_special_tokens=True))

        elif self.backend == "tiktoken":
            return len(self.tokenizer.encode(text))

        elif self.backend == "koboldcpp_api":
            return len(self._api_tokenize(text))

        return 0

    def tokenize(self, text: str) -> List[int]:
        """Tokenize text to list of token IDs."""
        if not text or not text.strip():
            return []

        if self.backend == "huggingface":
            return self.tokenizer.encode(text, add_special_tokens=True)

        elif self.backend == "tiktoken":
            return self.tokenizer.encode(text)

        elif self.backend == "koboldcpp_api":
            return self._api_tokenize(text)

        return []

    def detokenize(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text."""
        if not token_ids:
            return ""

        if self.backend == "huggingface":
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)

        elif self.backend == "tiktoken":
            return self.tokenizer.decode(token_ids)

        elif self.backend == "koboldcpp_api":
            return self._api_detokenize(token_ids)

        return ""

    def _api_tokenize(self, text: str, chunk_size: int = 45000) -> List[int]:
        """Tokenize using KoboldCpp API with chunking for large texts."""
        all_token_ids = []
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

        for chunk in chunks:
            try:
                response = requests.post(
                    f"{self.api_url.rstrip('/')}/api/extra/tokencount",
                    json={"prompt": chunk},
                    headers={"Content-Type": "application/json"},
                    timeout=9999
                )

                if response.status_code == 200:
                    data = response.json()
                    if "ids" in data:
                        all_token_ids.extend(data["ids"])
                    else:
                        raise RuntimeError("API tokencount returned no token IDs")
                else:
                    raise RuntimeError(f"API tokencount failed: {response.status_code}")

            except Exception as e:
                raise RuntimeError(f"API tokenization error: {e}")

        return all_token_ids

    def _api_detokenize(self, token_ids: List[int], chunk_size: int = 45000) -> str:
        """Detokenize using KoboldCpp API with chunking for large token arrays."""
        all_text = []
        chunks = [token_ids[i:i + chunk_size] for i in range(0, len(token_ids), chunk_size)]

        for chunk in chunks:
            try:
                response = requests.post(
                    f"{self.api_url.rstrip('/')}/api/extra/detokenize",
                    json={"ids": chunk},
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )

                if response.status_code == 200:
                    data = response.json()
                    if "result" in data:
                        all_text.extend(data["result"])
                    else:
                        raise RuntimeError("API detokenize returned no result")
                else:
                    raise RuntimeError(f"API detokenize failed: {response.status_code}")

            except Exception as e:
                raise RuntimeError(f"API detokenization error: {e}")

        return "".join(all_text)
