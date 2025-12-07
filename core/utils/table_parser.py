# core/utils/table_parser.py
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
from loguru import logger

from core.llm_client import LLMClient
from core.utils.parse_utils import normalize_table_for_llm

class LLMTableParser:
    def __init__(self):
        self.llm = LLMClient()
        self.max_rows_per_chunk = int(os.getenv("EXTRACT_MAX_ROWS_PER_CHUNK", "5"))
        self.workers = int(os.getenv("EXTRACT_LLM_WORKERS", "2"))

        logger.info(f"[LLM Table Parser] Initialized with Chunk Size: {self.max_rows_per_chunk}, Workers: {self.workers}")

    def _build_chunk_prompt(self, rows: List[List[str]], page_hint: int) -> str:
        csv_block = "\n".join([" | ".join(str(c) for c in r) for r in rows])

        return f"""
        You are a Data Engineer. Extract construction tasks from this table fragment.
        
        Context:
        - Page Hint: {page_hint}
        - Columns: ID | Task Name | Duration | Start | Finish

        Instructions:
        1. Return ONLY valid JSON with a key 'tasks'.
        2. Schema: {{ "task_id": int, "task_name": str, "duration_days": int, "start_date": "YYYY-MM-DD", "finish_date": "YYYY-MM-DD" }}
        3. Skip rows where ID is empty or not a number.
        4. If duration is empty, use 0. If dates are invalid, use null.

        Table Data:
        {csv_block}
        """

    def parse_table_hybrid(self, raw_rows: List[List[str]], page_num: int = 1) -> List[Dict]:
        normalized = normalize_table_for_llm(raw_rows)

        if not normalized:
            return []

        # Create chunks based on the configured size (now 5)
        chunks = [normalized[i:i + self.max_rows_per_chunk] for i in range(0, len(normalized), self.max_rows_per_chunk)]

        all_tasks = []
        logger.info(f"[Table Parser] Processing {len(chunks)} chunks for page {page_num}...")

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            future_to_chunk = {}
            for idx, chunk in enumerate(chunks):
                prompt = self._build_chunk_prompt(chunk, page_num)
                future = executor.submit(self.llm.parse_table_chunk, prompt=prompt)
                future_to_chunk[future] = idx

            for future in as_completed(future_to_chunk):
                try:
                    result = future.result()
                    tasks = result.get("tasks", [])
                    if tasks:
                        all_tasks.extend(tasks)
                except Exception as e:
                    logger.error(f"[LLM-Parser] Chunk {future_to_chunk[future]} failed: {e}")

        return all_tasks