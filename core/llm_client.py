import os
import json
import re
import ast
import requests
from loguru import logger
from typing import List, Dict, Any, Optional

class LLMClient:
    """Wrapper for Local LLMs with 'Silver Bullet' JSON Parsing."""

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        default_url = os.getenv("EXTRACT_LLM_ENDPOINT", "http://localhost:1234/v1/chat/completions")
        default_model = os.getenv("EXTRACT_LLM_MODEL", "qwen2.5-7b-instruct")
        self.base_url = base_url or default_url
        self.model = model or default_model

    def _clean_and_extract_json(self, content: str) -> str:
        content = content.strip()

        # 1. Regex to find the JSON block
        match = re.search(r"(\{.*\}|\[.*\])", content, re.DOTALL)
        if match:
            content = match.group(0)

        return content

    def ask_json(self, prompt: str, temperature: float = 0.0, max_tokens: int = -1) -> dict:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a data extraction agent. Return ONLY valid JSON."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }

        try:
            resp = requests.post(self.base_url, json=body, timeout=120)
            resp.raise_for_status()
            raw = resp.json()

            if "choices" not in raw or not raw["choices"]:
                return {}

            content = raw["choices"][0]["message"]["content"]

            # --- ROBUST PARSING STRATEGY ---
            cleaned = self._clean_and_extract_json(content)

            # Attempt 1: Strict JSON
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass # Try next method

            # Attempt 2: Fix common newline issue in Local LLMs
            # (They often put real newlines inside strings, which JSON hates)
            try:
                # Naive fix: remove newlines that are likely inside strings
                # This is risky but often necessary for 7B models
                return json.loads(cleaned.replace('\n', ' '))
            except json.JSONDecodeError:
                pass

            # Attempt 3: The Silver Bullet (Python Eval)
            # Python's parser handles single quotes and trailing commas
            try:
                return ast.literal_eval(cleaned)
            except (ValueError, SyntaxError):
                pass

            # If all failed, log details
            logger.error("[LLM JSON] All parsing attempts failed.")
            logger.debug(f"Failed Content: {cleaned[:500]}...")
            return {}

        except Exception as e:
            logger.error(f"[LLM JSON] Call failed: {e}")
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