import os
import json
import re
import ast
import requests
from loguru import logger
from typing import List, Dict, Any, Optional

class LLMClient:

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        # --- Configuration for Local LLM (Fallback) ---
        default_local_url = os.getenv("EXTRACT_LLM_ENDPOINT", "http://localhost:1234/v1/chat/completions")
        default_local_model = os.getenv("EXTRACT_LLM_MODEL", "qwen2.5-vl-7b-instruct")
        self.local_base_url = base_url or default_local_url
        self.local_model = model or default_local_model

        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_base_url = "https://api.openai.com/v1/chat/completions"
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def _clean_and_extract_json(self, content: str) -> str:
        content = content.strip()
        # Regex to find the JSON block ({...} or [...])
        match = re.search(r"(\{.*\}|\[.*\])", content, re.DOTALL)
        if match:
            content = match.group(0)
        return content

    def _make_llm_request(self, url: str, model: str, headers: dict, prompt: str, temperature: float, max_tokens: int) -> Optional[str]:
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a data extraction agent. Return ONLY valid JSON."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }

        try:
            resp = requests.post(url, headers=headers, json=body, timeout=120)
            resp.raise_for_status()
            raw = resp.json()

            if "choices" not in raw or not raw["choices"]:
                return None

            return raw["choices"][0]["message"]["content"]

        except Exception as e:
            logger.warning(f"[LLM Client] Request to {model} ({url}) failed: {e}")
            return None

    def ask_json(self, prompt: str, temperature: float = 0.0, max_tokens: int = -1) -> dict:
        content = None
        used_source = None

        if self.openai_api_key:
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }
            content = self._make_llm_request(
                self.openai_base_url,
                self.openai_model,
                headers,
                prompt,
                temperature,
                max_tokens
            )
            if content:
                used_source = "OpenAI"

        if content is None:
            if self.openai_api_key:
                logger.info("OpenAI failed or unavailable. Falling back to Local LLM...")

            headers = {"Content-Type": "application/json"}
            content = self._make_llm_request(
                self.local_base_url,
                self.local_model,
                headers,
                prompt,
                temperature,
                max_tokens
            )
            if content:
                used_source = "Local"

        if content is None:
            logger.error("llm not available") # Exact log message requested
            return {}

        cleaned = self._clean_and_extract_json(content)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        try:
            return json.loads(cleaned.replace('\n', ' '))
        except json.JSONDecodeError:
            pass

        try:
            return ast.literal_eval(cleaned)
        except (ValueError, SyntaxError):
            pass

        logger.error(f"[LLM JSON] All parsing attempts failed for source: {used_source}")
        logger.debug(f"Failed Content: {cleaned[:500]}...")
        return {}

    def parse_table_chunk(self, prompt: Optional[str] = None, rows: Optional[List[List[str]]] = None, **kwargs) -> Dict[str, Any]:
        temp = kwargs.get('temperature', 0.0)
        tokens = kwargs.get('max_tokens', -1)

        if not prompt and rows:
            csv_block = "\n".join([", ".join(str(c) for c in r) for r in rows])
            prompt = f"Extract tasks to JSON: {{'tasks': [{{'task_id': int, 'task_name': str, ...}}]}}\nData:\n{csv_block}"

        if not prompt:
            return {"tasks": []}

        result = self.ask_json(prompt, temperature=temp, max_tokens=tokens)

        if isinstance(result, list): return {"tasks": result}
        if "tasks" in result: return result
        return {"tasks": []}