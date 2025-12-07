from typing import List, Dict, Any
from pipeline.schemas import TaskSchema
from core.utils.parse_utils import _parse_date_flexible  # Import your existing parser

def merge_tasks(task_list: List[Any]) -> List[TaskSchema]:
    """
    Accepts: list of TaskSchema or list of dicts or mixed.
    Returns: deduped list[TaskSchema].
    """
    final: Dict[int, Dict] = {}

    for t in task_list:

        # 1. Normalize input to Dict
        if isinstance(t, TaskSchema):
            # Support both Pydantic v1 and v2
            t = t.model_dump() if hasattr(t, "model_dump") else t.dict()
        elif not isinstance(t, dict):
            continue

        # 2. Extract ID
        tid = t.get("task_id")
        if not tid:
            continue

        # Ensure ID is int
        try:
            tid = int(tid)
        except (ValueError, TypeError):
            continue

        # 3. CRITICAL FIX: Normalize Dates from LLM strings
        # LLM might return '2007-09-5', we convert it to a Python date object here.
        if "start_date" in t and isinstance(t["start_date"], str):
            t["start_date"] = _parse_date_flexible(t["start_date"])

        if "finish_date" in t and isinstance(t["finish_date"], str):
            t["finish_date"] = _parse_date_flexible(t["finish_date"])

        # 4. Merge Logic
        if tid not in final:
            final[tid] = t
            continue

        # Strategy: Prefer the entry with the longer task_name (usually better detail)
        old = final[tid]
        new_name = str(t.get("task_name", ""))
        old_name = str(old.get("task_name", ""))

        if len(new_name) > len(old_name):
            final[tid] = t

        # Strategy: Prefer non-null dates if the existing one is null
        if not old.get("start_date") and t.get("start_date"):
            final[tid]["start_date"] = t["start_date"]
        if not old.get("finish_date") and t.get("finish_date"):
            final[tid]["finish_date"] = t["finish_date"]

        # Strategy: Prefer non-zero duration
        if not old.get("duration_days") and t.get("duration_days"):
            final[tid]["duration_days"] = t["duration_days"]

    # 5. Convert back to TaskSchema
    # Now safe because dates are python objects, not dirty strings
    return [TaskSchema(**v) for v in final.values()]