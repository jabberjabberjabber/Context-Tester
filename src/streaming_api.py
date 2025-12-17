#!/usr/bin/env python3
"""
API client for text generation using OpenAI-compatible endpoints.

Supports KoboldCpp, NVIDIA NIM, and generic OpenAI-compatible APIs.
"""

from typing import Optional, List
import json
import requests
import logging
from pathlib import Path
from datetime import datetime
from src.find_last_sentence import find_last_sentence_ending
from src.chunker_regex import chunk_regex
from src.tokenizer_utils import UnifiedTokenizer

# Setup logging
logger = logging.getLogger(__name__)

def setup_api_logging(log_dir: Optional[Path] = None, level: int = logging.DEBUG):
    """Setup detailed logging for API calls.

    Args:
        log_dir: Directory for log files (defaults to ./logs)
        level: Logging level (default: DEBUG)
    """
    if log_dir is None:
        log_dir = Path("logs")

    log_dir.mkdir(exist_ok=True)

    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"api_debug_{timestamp}.log"

    # Configure logger
    logger.setLevel(level)

    # File handler with detailed formatting
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    # Console handler with simpler formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Less verbose on console
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)

    # Add handlers if not already added
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    logger.info(f"API logging initialized - log file: {log_file}")
    return log_file


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
        hf_token: Optional[str] = None,
        enable_logging: bool = True,
        log_dir: Optional[Path] = None
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
            enable_logging: Enable detailed API logging (default: True)
            log_dir: Directory for log files (default: ./logs)
        """
        # Setup logging if enabled
        if enable_logging:
            self.log_file = setup_api_logging(log_dir)
            logger.info("="*80)
            logger.info("Initializing StreamingAPIClient")
            logger.info(f"API URL (raw): {api_url}")
        else:
            self.log_file = None

        self.api_url = api_url

        # Detect API type and set appropriate endpoint paths
        if 'z.ai' in self.api_url.lower() or '/api/paas/v4' in self.api_url:
            # Z.AI uses /api/paas/v4/chat/completions
            self.endpoint_path = '/api/paas/v4'
            self.embedding_model = embedding_model or "glm-4.6"
            if not self.api_url.endswith('/chat/completions'):
                self.api_url = f"{self.api_url.rstrip('/')}/chat/completions"
            logger.info("Detected Z.AI endpoint")
        else:
            # Standard OpenAI-compatible uses /v1/chat/completions
            self.endpoint_path = '/v1'
            self.embedding_model = embedding_model or "nvidia/nv-embed-v1"
            if not self.api_url.endswith('/v1/chat/completions'):
                self.api_url = f"{self.api_url.rstrip('/')}/v1/chat/completions"
            logger.info("Using standard OpenAI-compatible endpoint")
            

        logger.info(f"API URL (normalized): {self.api_url}")
        logger.info(f"Endpoint path: {self.endpoint_path}")

        self.headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        if api_password:
            self.headers["Authorization"] = f"Bearer {api_password}"
            # Log sanitized version only
            logger.info(f"API key configured: {api_password[:4]}...{api_password[-4:]}")
        else:
            logger.warning("No API key provided")

        self.model_name = model_name

        self._max_context = max_context

        logger.info(f"Model name: {model_name}")
        logger.info(f"Tokenizer model: {tokenizer_model}")
        logger.info(f"Max context: {max_context}")

        # Initialize unified tokenizer
        base_url = self._get_base_url()
        logger.info(f"Base URL: {base_url}")

        self.tokenizer = UnifiedTokenizer(
            api_url=base_url,
            api_headers=self.headers,
            tokenizer_model=tokenizer_model,
            model_name=model_name,
            hf_token=hf_token
        )
        logger.info("StreamingAPIClient initialized successfully")
        logger.info("="*80)

        # Print log file location to console
        if self.log_file:
            print(f"\n📝 API Debug Logs: {self.log_file}")
            print(f"    Check this file for detailed request/response information\n")

    def get_log_file(self) -> Optional[Path]:
        """Get the path to the current log file."""
        return self.log_file

    def _get_base_url(self) -> str:
        """Get base URL without endpoint-specific paths.

        Returns:
            Base URL with endpoint path stripped
        """
        if self.endpoint_path == '/api/paas/v4':
            # Z.AI endpoint
            return self.api_url.replace('/api/paas/v4/chat/completions', '').replace('/chat/completions', '')
        else:
            # Standard OpenAI-compatible endpoint
            return self.api_url.replace('/v1/chat/completions', '')

    def is_kobold_api(self) -> bool:
        """Check if the API is a KoboldCpp instance.

        Returns:
            True if this is a KoboldCpp API, False otherwise
        """
        try:
            # Get base URL without endpoint paths
            base_url = self._get_base_url()
            version_url = f"{base_url.rstrip('/')}/api/extra/version"

            # Try to get version info (KoboldCpp-specific endpoint)
            response = requests.get(version_url, timeout=5)

            if response.status_code == 200:
                try:
                    version_data = response.json()
                    # Check if response has KoboldCpp-specific fields
                    if 'result' in version_data or 'version' in version_data:
                        return True
                except (json.JSONDecodeError, ValueError):
                    pass

            return False

        except (requests.exceptions.RequestException, requests.exceptions.Timeout):
            # If request fails, assume not KoboldCpp
            return False

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
            logger.warning("get_embeddings called with empty texts list")
            return None

        model = model or self.embedding_model
        logger.info(f"Getting embeddings for {len(texts)} text(s) using model: {model}")

        # Z.AI has a limit of 64 texts per request, batch if needed
        batch_size = 64 if 'z.ai' in self.api_url.lower() else 2048

        if len(texts) > batch_size:
            logger.info(f"Batching {len(texts)} texts into chunks of {batch_size}")
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size} ({len(batch)} texts)")
                batch_embeddings = self._get_embeddings_batch(batch, model)
                if batch_embeddings is None:
                    logger.error(f"Batch {i//batch_size + 1} failed")
                    return None
                all_embeddings.extend(batch_embeddings)
            logger.info(f"Successfully retrieved {len(all_embeddings)} embeddings across all batches")
            return all_embeddings
        else:
            return self._get_embeddings_batch(texts, model)

    def _get_embeddings_batch(self, texts: List[str], model: str) -> Optional[List[List[float]]]:
        """Get embeddings for a single batch of texts.

        Args:
            texts: List of strings to embed (should be <= batch_size)
            model: Embedding model name

        Returns:
            List of embedding vectors, or None if API call fails
        """
        try:
            base_url = self._get_base_url()
            # Fix double slash issue: ensure base_url doesn't end with /
            base_url = base_url.rstrip('/')
            embeddings_url = f"{base_url}{self.endpoint_path}/embeddings"

            payload = {
                "input": texts,
                "model": model,
                "encoding_format": "float",
                "truncate": "NONE"
            }

            response = self._make_request_with_retry(embeddings_url, payload=payload)

            if response is None:
                logger.error("Embeddings request returned None")
                return None

            if response.status_code == 200:
                data = response.json()
                if "data" in data:
                    embeddings = sorted(data["data"], key=lambda x: x.get("index", 0))
                    result = [item["embedding"] for item in embeddings]
                    logger.info(f"Successfully retrieved {len(result)} embeddings")
                    return result
                else:
                    logger.error("No 'data' field in API response")
                    logger.error(f"Response: {json.dumps(data, indent=2)[:500]}")
                    print("No embedding data in API response")
                    return None
            else:
                logger.error(f"Embeddings API error {response.status_code}: {response.text}")
                print(f"Embeddings API error {response.status_code}: {response.text[:200]}")
                return None

        except Exception as e:
            logger.exception(f"Error getting embeddings: {e}")
            print(f"Error getting embeddings: {e}")
            return None
    
    def get_max_context_length(self) -> int:
        """Get model's maximum context length from API or configuration."""
        # Use configured value if available
        if self._max_context:
            return self._max_context

        # Try to detect from API (KoboldCpp mode)
        try:
            base_url = self._get_base_url()
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
            base_url = self._get_base_url()
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

        logger.info("="*80)
        logger.info(f"Making API request to: {url}")
        logger.debug(f"Stream mode: {stream}")
        logger.debug(f"Max retries: {max_retries}")

        # Log payload (sanitize sensitive data) - use deep copy to avoid modifying actual payload
        import copy
        sanitized_payload = copy.deepcopy(payload)
        if 'messages' in sanitized_payload and isinstance(sanitized_payload['messages'], list):
            for msg in sanitized_payload['messages']:
                if 'content' in msg and len(str(msg['content'])) > 200:
                    msg['content'] = str(msg['content'])[:200] + f"... [truncated {len(str(msg['content']))-200} chars]"

        logger.debug(f"Request payload: {json.dumps(sanitized_payload, indent=2)}")

        # Log headers (hide authorization)
        safe_headers = self.headers.copy()
        if 'Authorization' in safe_headers:
            auth = safe_headers['Authorization']
            if auth.startswith('Bearer '):
                token = auth[7:]
                safe_headers['Authorization'] = f"Bearer {token[:4]}...{token[-4:]}"

        logger.debug(f"Request headers: {json.dumps(safe_headers, indent=2)}")

        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries}")

                response = requests.post(
                    url,
                    json=payload,
                    headers=self.headers,
                    stream=stream,
                    timeout=9999
                )

                logger.info(f"Response status code: {response.status_code}")
                logger.debug(f"Response headers: {dict(response.headers)}")

                if response.status_code == 200:
                    logger.info("Request successful (200 OK)")
                    if not stream:
                        try:
                            response_json = response.json()
                            logger.debug(f"Response JSON: {json.dumps(response_json, indent=2)[:1000]}")
                        except:
                            logger.debug(f"Response text (first 500 chars): {response.text[:500]}")
                    logger.info("="*80)
                    return response

                elif response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limited (429). Retry-After: {retry_after}s")
                    logger.warning(f"Response: {response.text[:500]}")
                    print(f"Rate limited. Waiting {retry_after}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(retry_after)

                elif response.status_code == 400:
                    logger.error(f"Bad request (400): {response.text}")
                    try:
                        error_json = response.json()
                        logger.error(f"Error details: {json.dumps(error_json, indent=2)}")
                    except:
                        pass
                    print(f"Request failed: Bad request (400)")
                    logger.info("="*80)
                    return None

                elif response.status_code == 401:
                    logger.error(f"Unauthorized (401) - Check API key")
                    logger.error(f"Response: {response.text}")
                    print(f"Request failed: Unauthorized (401) - Check your API key")
                    logger.info("="*80)
                    return None

                elif response.status_code == 404:
                    logger.error(f"Not found (404) - Check endpoint URL")
                    logger.error(f"URL: {url}")
                    logger.error(f"Response: {response.text}")
                    print(f"Request failed: Endpoint not found (404)")
                    logger.info("="*80)
                    return None

                else:
                    logger.error(f"Request failed with status {response.status_code}")
                    logger.error(f"Response text: {response.text}")
                    try:
                        error_json = response.json()
                        logger.error(f"Error JSON: {json.dumps(error_json, indent=2)}")
                    except:
                        pass
                    print(f"Request failed with status {response.status_code}: {response.text[:200]}")
                    logger.info("="*80)
                    return None

            except requests.exceptions.Timeout as e:
                logger.error(f"Request timeout on attempt {attempt + 1}: {e}")
                print(f"Request timeout (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    sleep_time = 2 ** attempt
                    logger.info(f"Retrying after {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    logger.error("Max retries exceeded (timeout)")
                    logger.info("="*80)
                    return None

            except requests.exceptions.ConnectionError as e:
                logger.error(f"Connection error on attempt {attempt + 1}: {e}")
                print(f"Connection error (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    sleep_time = 2 ** attempt
                    logger.info(f"Retrying after {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    logger.error("Max retries exceeded (connection error)")
                    logger.info("="*80)
                    return None

            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {type(e).__name__}: {e}")
                logger.exception("Full traceback:")
                print(f"Request error: {e}")
                if attempt < max_retries - 1:
                    sleep_time = 2 ** attempt
                    logger.info(f"Retrying after {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    logger.error("Max retries exceeded (unexpected error)")
                    logger.info("="*80)
                    return None

        logger.error(f"Max retries ({max_retries}) exceeded")
        print(f"Max retries exceeded")
        logger.info("="*80)
        return None

    def generate_continuation(
        self,
        context: str,
        max_tokens: int = 1024,
        temperature: float = 0.1,
        top_k: int = None,
        top_p: float = None,
        no_think: bool = False,
        seed = None
    ) -> str:
        """Generate text continuation from context."""
        instruction = """This is an important test of your ability to write long form narratives when burdened with a rich text as starting point. Continue this story without moving towards any conclusions. Continue to develop characters, motivations, world, story, and plot from the text. Maintain the same style, tone, voice, structure, syntax and verbal flourish of the author but strive for diversity, complexity, and creativity. Do not reference these instructions nor ruminate. Begin writing."""


        if not context:
            return None

        context = find_last_sentence_ending(context)
        print(f"\n\n...{context[-250:]}\n\n...")
        payload = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "n": 1
        }
        if top_p:
            payload["top_p"] = top_p
        if seed:
            payload["seed"] = seed
        if top_k:
            payload["top_k"] = top_k
        # Handle thinking/reasoning settings based on model type
        if "glm" in self.model_name.lower():
            # Z.AI GLM models - use their specific thinking format
            payload["messages"] = [{"role": "user", "content": f"{context}\n\n{instruction}"}]
            if no_think:
                payload["thinking"] = {"type": "disabled"}
            else:
                payload["thinking"] = {"type": "enabled"}
        elif no_think and "deepseek" in self.model_name.lower():
            # DeepSeek reasoning models with thinking disabled
            payload["messages"] = [
                        {"role": "system", "content": "/no_think"},
                        {"role": "user", "content": f"{context}\n\n{instruction}"}
            ]
            payload["extra_body"]={
                "chat_template_kwargs": {"thinking": False},
                "separate_reasoning": False
            }
        elif no_think and "qwen" in self.model_name.lower():
            # Qwen reasoning models with thinking disabled
            payload["messages"] = [
                        {"role": "system", "content": "/no_think"},
                        {"role": "user", "content": f"{context}\n\n{instruction}"}
            ]
            payload["extra_body"]={
                "chat_template_kwargs": {"enable_thinking": False},
                "separate_reasoning": False
            }
        else:
            # Standard models without thinking (includes NVIDIA, Gemma, Mistral, etc.)
            # Don't send extra_body parameters as most providers don't support them
            payload["messages"] = [{"role": "user", "content": f"{context}\n\n{instruction}"}]       


                
        result = []

        try:
            response = self._make_request_with_retry(
                self.api_url,
                payload=payload,
                stream=True
            )

            if not response:
                return ""

            logger.info("Processing streaming response...")
            line_count = 0
            content_chunks = 0

            for line in response.iter_lines():
                if not line:
                    continue

                line_count += 1
                line_text = line.decode('utf-8').strip()

                # Log first few lines to see format
                if line_count <= 3:
                    logger.debug(f"Stream line {line_count}: {line_text[:200]}")

                # Handle SSE format (data: prefix)
                if line_text.startswith('data: '):
                    line_text = line_text[6:].strip()

                # Skip empty lines and done markers
                if not line_text or line_text == '[DONE]':
                    if line_text == '[DONE]':
                        logger.info("Received [DONE] marker")
                    continue

                try:
                    data = json.loads(line_text)

                    # Log structure of first response
                    if line_count == 1:
                        logger.debug(f"First chunk structure: {json.dumps(data, indent=2)[:500]}")

                    # Try multiple formats for compatibility
                    content = None

                    # Standard OpenAI format: delta.content
                    if 'choices' in data and len(data['choices']) > 0:
                        choice = data['choices'][0]

                        # Format 1: delta.content (standard streaming)
                        if 'delta' in choice and 'content' in choice['delta']:
                            content = choice['delta']['content']

                        # Format 2: message.content (some providers use this)
                        elif 'message' in choice and 'content' in choice['message']:
                            content = choice['message']['content']

                        # Format 3: text field (alternative format)
                        elif 'text' in choice:
                            content = choice['text']

                        # Format 4: Z.AI reasoning models - skip reasoning_content, wait for content
                        # (reasoning_content is ignored, only actual content is captured)

                    if content is not None and content != "":
                        result.append(content)
                        content_chunks += 1
                        print(content, end='', flush=True)

                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    # Log parsing errors for debugging
                    if line_count <= 5:
                        logger.warning(f"Failed to parse stream line {line_count}: {type(e).__name__}: {e}")
                        logger.debug(f"Problematic line: {line_text[:200]}")
                    continue

            logger.info(f"Streaming complete: {line_count} lines, {content_chunks} content chunks, {len(result)} total chars")
            print()

            if not result:
                logger.error("No content received from streaming response!")
                return ""

            return ''.join(result)

        except Exception as e:
            logger.exception(f"Error in generation: {str(e)}")
            print(f"\nError in generation: {str(e)}")
            return ""

    def generate_continuation_litellm(
        self,
        context: str,
        max_tokens: int = 1024,
        temperature: float = 0.1,
        top_k: int = None,
        top_p: float = None,
        no_think: bool = False,
        seed = None
    ) -> str:
        """Generate text continuation using LiteLLM for universal compatibility.

        LiteLLM provides a unified interface to various LLM providers and handles
        provider-specific quirks automatically.

        Args:
            context: Input text to continue
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            no_think: Disable thinking for reasoning models
            seed: Random seed for reproducibility

        Returns:
            Generated continuation text
        """
        try:
            from litellm import completion
            import os
        except ImportError:
            print("Warning: litellm not installed. Install with: pip install litellm")
            return ""

        instruction = """This is an important test of your ability to write long form narratives when burdened with a rich text as starting point. Continue this story without moving towards any conclusions. Continue to develop characters, motivations, world, story, and plot from the text. Maintain the same style, tone, voice, structure, syntax and verbal flourish of the author but strive for diversity, complexity, and creativity. Do not reference these instructions nor ruminate. Begin writing."""

        if not context:
            return None

        context = find_last_sentence_ending(context)
        print(f"\n\n...{context[-250:]}\n\n...")

        # Build messages
        if no_think and self.model_name and ("gemma" not in self.model_name):
            messages = [
                {"role": "system", "content": "/no_think"},
                {"role": "user", "content": f"{context}\n\n{instruction}"}
            ]
        else:
            messages = [{"role": "user", "content": f"{context}\n\n{instruction}"}]

        # Prepare LiteLLM parameters
        litellm_params = {
            "model": f"openai/{self.model_name}" if self.model_name else "openai/gpt-3.5-turbo",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

        # Add optional parameters
        if top_p is not None:
            litellm_params["top_p"] = top_p
        if seed is not None:
            litellm_params["seed"] = seed
        if top_k is not None:
            litellm_params["top_k"] = top_k

        # Configure custom API endpoint
        base_url = self._get_base_url()
        litellm_params["api_base"] = base_url

        # Set API key if provided
        if self.headers.get("Authorization"):
            # Extract bearer token
            auth_header = self.headers["Authorization"]
            if auth_header.startswith("Bearer "):
                api_key = auth_header[7:]  # Remove "Bearer " prefix
                litellm_params["api_key"] = api_key
                # Also set as environment variable for LiteLLM
                os.environ["OPENAI_API_KEY"] = api_key

        # Add extra_body for reasoning models (only for known compatible models)
        if no_think and self.model_name:
            if "deepseek" in self.model_name.lower():
                litellm_params["extra_body"] = {
                    "chat_template_kwargs": {"thinking": False},
                    "separate_reasoning": False
                }
            elif "qwen" in self.model_name.lower():
                litellm_params["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": False},
                    "separate_reasoning": False
                }
            # Don't add extra_body for other models (Mistral, etc.) as they don't support it

        result = []

        try:
            print(f"Using LiteLLM for generation...")

            response = completion(**litellm_params)

            # Stream the response
            for chunk in response:
                if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content is not None:
                        content = delta.content
                        result.append(content)
                        print(content, end='', flush=True)

            print()
            return ''.join(result)

        except Exception as e:
            print(f"\nError in LiteLLM generation: {str(e)}")
            import traceback
            traceback.print_exc()
            return ""

    def generate_continuation_with_fallback(
        self,
        context: str,
        max_tokens: int = 1024,
        temperature: float = 0.1,
        top_k: int = None,
        top_p: float = None,
        no_think: bool = False,
        seed = None,
        use_litellm: bool = False
    ) -> str:
        """Generate text continuation with automatic fallback to LiteLLM.

        Tries the direct API method first. If that fails, automatically falls back
        to using LiteLLM for broader compatibility.

        Args:
            context: Input text to continue
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            no_think: Disable thinking for reasoning models
            seed: Random seed for reproducibility
            use_litellm: Force use of LiteLLM instead of trying direct API first

        Returns:
            Generated continuation text
        """
        # If explicitly requested to use LiteLLM, skip the direct API
        if use_litellm:
            print("Using LiteLLM (explicitly requested)...")
            return self.generate_continuation_litellm(
                context=context,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                no_think=no_think,
                seed=seed
            )

        # Try direct API first
        print("Attempting direct API call...")
        result = self.generate_continuation(
            context=context,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            no_think=no_think,
            seed=seed
        )

        # If direct API succeeded, return result
        if result and len(result.strip()) > 0:
            return result

        # Direct API failed, try LiteLLM fallback
        print("\nDirect API failed or returned empty result. Falling back to LiteLLM...")
        return self.generate_continuation_litellm(
            context=context,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            no_think=no_think,
            seed=seed
        )
