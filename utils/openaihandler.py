"""OpenAI API Handler with multi-threading and key rotation."""

import os
import time
import threading
from typing import List, Optional, Dict
from queue import Queue
import random
import requests


class OpenAIHandler:
    """Handler for calling OpenAI-compatible APIs with multi-threading and key rotation."""

    def __init__(
        self,
        api_keys: List[str],
        api_base: str,
        model_name: str,
        max_workers: int = 64,
        max_retries: int = 10,
        retry_delay: float = 1.0,
        proxies: Optional[Dict[str, str]] = None,
    ):
        """Initialize OpenAI Handler.

        Args:
            api_keys: List of API keys to rotate through
            api_base: Base URL for the API
            model_name: Model name to use
            max_workers: Maximum number of parallel workers
            max_retries: Maximum number of retries per request
            retry_delay: Delay between retries in seconds
            proxies: Optional proxy configuration for requests (e.g., {"http": "...", "https": "..."})
        """
        self.api_keys = api_keys
        self.api_base = api_base
        self.model_name = model_name
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.proxies = proxies

        # Thread-safe key rotation
        self.key_lock = threading.Lock()
        self.current_key_idx = 0

    def _get_next_api_key(self) -> str:
        """Get next API key in round-robin fashion (thread-safe)."""
        with self.key_lock:
            key = self.api_keys[self.current_key_idx]
            self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
            return key

    def _call_api_with_retry(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        enable_thinking: bool = False,
    ) -> str:
        """Call API with infinite retry until success, rotating keys on failure.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        retry_count = 0
        last_exception = None

        # Construct API endpoint
        api_url = f"{self.api_base.rstrip('/')}/chat/completions"

        while True:
            # Get next key for this attempt
            api_key = self._get_next_api_key()

            try:
                # Prepare headers
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }

                # Prepare request body
                data = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "enable_thinking": enable_thinking,
                }

                # Make API call with requests
                response = requests.post(
                    api_url,
                    headers=headers,
                    json=data,
                    proxies=self.proxies,
                    timeout=300,  # 5 minutes timeout
                )

                # Check if request was successful
                response.raise_for_status()

                # Parse response
                response_json = response.json()
                message = response_json["choices"][0]["message"]
                generated_text = message.get("content")
                if generated_text is None:
                    generated_text = message.get("reasoning_content")
                    if generated_text is None:
                        generated_text = ""
                    else:
                        generated_text = str(generated_text)
                        if "</think>" in generated_text:
                            generated_text = generated_text.split("</think>", 1)[1]
                        generated_text = generated_text.strip()
                elif not isinstance(generated_text, str):
                    generated_text = str(generated_text)
                return generated_text

            except Exception as e:
                last_exception = e
                retry_count += 1

                print(
                    f"[OpenAIHandler] API call failed (attempt {retry_count}): {e}. "
                    f"Retrying with next key after {self.retry_delay}s..."
                )

                # Wait before retry
                time.sleep(self.retry_delay)

                # Continue to next iteration (infinite retry)
                continue

    def batch_generate(
        self,
        prompts: List[str],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        enable_thinking: bool = False,
    ) -> List[str]:
        """Generate responses for multiple prompts in parallel.

        Args:
            prompts: List of input prompts
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            List of generated texts (same order as input prompts)
        """
        if len(prompts) == 0:
            return []

        # Results storage (indexed by prompt index)
        results = [None] * len(prompts)
        results_lock = threading.Lock()

        # Task queue
        task_queue = Queue()
        for idx, prompt in enumerate(prompts):
            task_queue.put((idx, prompt))

        # Worker function
        def worker():
            while True:
                try:
                    # Get task (non-blocking with timeout)
                    idx, prompt = task_queue.get(timeout=0.1)
                except:
                    # Queue is empty, exit
                    break

                try:
                    # Call API with retry
                    generated_text = self._call_api_with_retry(
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        enable_thinking=enable_thinking,
                    )

                    # Store result (thread-safe)
                    with results_lock:
                        results[idx] = generated_text

                finally:
                    # Mark task as done
                    task_queue.task_done()

        # Create and start worker threads
        threads = []
        num_workers = min(self.max_workers, len(prompts))

        print(f"[OpenAIHandler] Starting {num_workers} workers to process {len(prompts)} prompts...")

        for _ in range(num_workers):
            t = threading.Thread(target=worker)
            t.start()
            threads.append(t)

        # Wait for all tasks to complete
        task_queue.join()

        # Wait for all threads to finish
        for t in threads:
            t.join()

        print(f"[OpenAIHandler] All {len(prompts)} prompts processed successfully.")

        return results


def load_api_keys_from_file(file_path: str) -> List[str]:
    """Load API keys from file, one per line, ignoring empty lines.

    Args:
        file_path: Path to API keys file

    Returns:
        List of API keys
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"API keys file not found: {file_path}")

    api_keys = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Ignore empty lines
                api_keys.append(line)

    if len(api_keys) == 0:
        raise ValueError(f"No API keys found in file: {file_path}")

    print(f"[OpenAIHandler] Loaded {len(api_keys)} API keys from {file_path}")
    return api_keys
