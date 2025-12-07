# core/table_parser.py
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional
from loguru import logger

from pipeline.schemas import TaskSchema
from core.llm_client import LLMClient
# Ensure this import matches the file above
from core.utils.parse_utils import normalize_table_for_llm

class LLMTableParser:
    def __init__(self):
        self.llm = LLMClient()
        self.max_rows_per_chunk = 25
        self.workers = int(os.getenv("EXTRACT_LLM_WORKERS", "4"))

    def _build_chunk_prompt(self, rows: List[List[str]], page_hint: int) -> str:
        """Constructs the prompt with strict constraints against hallucinations."""
        csv_block = "\n".join([" | ".join(str(c) for c in r) for r in rows])

        return f"""
        You are a Data Engineer. Extract construction tasks from this table fragment.
        
        Context:
        - Columns: ID | Task Name | Duration | Start | Finish

        Instructions:
        1. Return ONLY valid JSON with a key 'tasks'.
        2. Schema: {{ "task_id": int, "task_name": str, "duration_days": int, "start_date": "YYYY-MM-DD", "finish_date": "YYYY-MM-DD" }}
        3. CRITICAL: Do NOT merge multiple rows into one task. Keep task_name short and precise.
        4. If a row has multiple unrelated concepts, split them or pick the main one.
        5. Skip rows where ID is empty or not a number.

        Table Data:
        {csv_block}
        """

    def parse_table_hybrid(self, raw_rows: List[List[str]], page_num: int = 1) -> List[Dict]:
        """
        Main entry point. Normalizes table, chunks it, and processes with LLM.
        """
        # This call was failing because _clean_text was missing in parse_utils
        normalized = normalize_table_for_llm(raw_rows)

        if not normalized:
            return []

        chunks = [normalized[i:i + self.max_rows_per_chunk] for i in range(0, len(normalized), self.max_rows_per_chunk)]
        all_tasks = []

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            future_to_chunk = {}
            for idx, chunk in enumerate(chunks):
                prompt = self._build_chunk_prompt(chunk, page_num)
                future = executor.submit(self.llm.parse_table_chunk, prompt=prompt)
                future_to_chunk[future] = idx

            for future in as_completed(future_to_chunk):
                try:
                    result = future.result() # Returns {"tasks": [...]}
                    tasks = result.get("tasks", [])
                    if tasks:
                        all_tasks.extend(tasks)
                except Exception as e:
                    logger.error(f"[LLM-Parser] Chunk {future_to_chunk[future]} failed: {e}")

        return all_tasks