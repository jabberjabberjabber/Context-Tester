from typing import Dict, List, Optional, Set, Any, Literal, TypeAlias
import json
import requests
from find_last_sentence import find_last_sentence_ending

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
    
    def get_embeddings(self, texts: List[str], model: str = "text-embedding-ada-002") -> Optional[List[List[float]]]:
        """ Get embeddings for a list of texts using OpenAI-compatible API
        
        Args:
            texts: List of strings to embed
            model: Embedding model name (API may ignore this)
        
        Returns:
            List of embedding vectors, or None if API call fails
        """
        if not texts:
            return None
            
        try:
            # Use same base URL but with embeddings endpoint
            base_url = self.api_url.replace('/v1/chat/completions', '')
            embeddings_url = f"{base_url}/v1/embeddings"
            
            payload = {
                "input": texts,
                "model": model
            }
            
            response = requests.post(
                embeddings_url,
                json=payload,
                headers={"Content-Type": "application/json", **{k: v for k, v in self.headers.items() if k == "Authorization"}},
                timeout=60
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
        base_url = self.api_url.replace('v1/chat/completions', '')
        try:
            response = requests.post(
                f"{base_url}/api/extra/tokencount",
                json={"prompt": text},
                headers={"Content-Type": "application/json"},
                timeout=180
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
        
        Uses binary search over chunk boundaries to find the largest amount
        of text that fits within max_context tokens when chunked naturally.
        """
        
        total_tokens = self.count_tokens(text)
        if total_tokens < max_context:
            print(f"Full text ({total_tokens:,} tokens) fits within max context ({max_context:,})")
            return text

        print(f"Pruning text from {total_tokens:,} tokens to fit {max_context:,} tokens at natural boundaries...")
        
        # Get all chunk boundaries
        matches = list(chunk_regex.finditer(text))
        if not matches:
            print("Warning: No regex matches found, using character-based truncation")
            # Fallback: truncate at max_context * 4 characters (rough estimate)
            return text[:max_context * 4]
        
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
                    timeout=9999
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
        context = find_last_sentence_ending(context)
        print(f"Starting from: {context[-100:]}")
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
                timeout=999
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