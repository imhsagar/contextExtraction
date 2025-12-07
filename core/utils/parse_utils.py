# core/utils/parse_utils.py
import re
from typing import List, Any, Dict, Optional
from datetime import datetime
from loguru import logger
from pipeline.schemas import TaskSchema

# ============================================================
# 1. TEXT CLEANING UTILITIES
# ============================================================

def _clean_text(s: Optional[str]) -> str:
    """Normalize whitespace and remove weird characters."""
    if not s:
        return ""
    s = str(s).replace("\n", " ").replace("\t", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _clean_task_name(name: Optional[str]) -> Optional[str]:
    """Reject impossible task names and sanitize."""
    if not name:
        return None
    name = _clean_text(name)
    # Reject extremely long garbage task names
    if len(name) > 200: return None
    # Reject names with mostly numbers
    if re.fullmatch(r"[\d\W]+", name): return None
    # Reject header-like text
    if name.lower() in ["task name", "task", "name", "activity", "description"]: return None
    return name

# ============================================================
# 2. PARSING HELPERS (Dates, Ints, Durations)
# ============================================================

def _parse_int_safe(x) -> Optional[int]:
    """Parse ID-only integers."""
    if x is None: return None
    s = str(x).strip()
    if not re.fullmatch(r"\d+", s): return None
    val = int(s)
    if val > 99999: return None # Reject merged cells
    return val

def _parse_duration(s: Optional[str]) -> int:
    """Extract integer days from string."""
    if not s: return 0
    s = str(s).lower().strip()
    # Match "10 d", "10 days", or just "10"
    m = re.search(r"(\d+)\s*d", s) or re.search(r"(\d+)", s)
    if m:
        try:
            return int(m.group(1))
        except:
            return 0
    return 0

DATE_FORMATS = [
    "%d-%b-%y", "%d-%b-%Y", "%Y-%m-%d",
    "%m/%d/%y", "%m/%d/%Y", "%d.%m.%y"
]

def _parse_date_flexible(s: Optional[str]):
    if not s: return None
    s = _clean_text(s)
    # Basic cleanup for OCR noise
    s = s.replace("|", "").replace("l", "1").strip()

    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except:
            continue
    return None

# ============================================================
# 3. DATA CONVERSION & NORMALIZATION
# ============================================================

def safe_task_to_dict(t: Any) -> Dict:
    """Convert TaskSchema or dict to plain dict."""
    if t is None: return {}
    if isinstance(t, dict): return t
    if hasattr(t, "model_dump"): return t.model_dump()
    if hasattr(t, "dict"): return t.dict()

    out = {}
    for k in ["task_id", "task_name", "duration_days", "start_date", "finish_date"]:
        if hasattr(t, k):
            val = getattr(t, k)
            if isinstance(val, datetime): val = val.date().isoformat()
            out[k] = val
    return out

def normalize_table_for_llm(table_rows: List[List[str]]) -> List[List[str]]:
    """Reduces raw table to critical columns for LLM."""
    # Ensure _clean_text is available here
    rows = [[_clean_text(c) for c in row] for row in table_rows]

    # Heuristic: Find header
    header = None
    header_idx = 0
    for i, r in enumerate(rows[:5]):
        combined = " ".join(r).lower()
        if "id" in combined and ("task" in combined or "description" in combined):
            header = r
            header_idx = i
            break

    if header:
        col_map = {"id": None, "task": None, "dur": None, "start": None, "end": None}
        for idx, h in enumerate(header):
            lh = h.lower()
            if "id" in lh: col_map["id"] = idx
            elif "task" in lh or "activity" in lh: col_map["task"] = idx
            elif "dur" in lh: col_map["dur"] = idx
            elif "start" in lh: col_map["start"] = idx
            elif "finish" in lh or "end" in lh: col_map["end"] = idx

        if col_map["id"] is not None and col_map["task"] is not None:
            indices = [col_map.get(k) for k in ["id", "task", "dur", "start", "end"]]
            norm = []
            for r in rows[header_idx+1:]:
                new_row = []
                for i in indices:
                    if i is not None and i < len(r): new_row.append(r[i])
                    else: new_row.append("")
                norm.append(new_row)
            return norm

    # Fallback
    return [r[:5] for r in rows]

# ============================================================
# 4. RULE-BASED PARSING LOGIC (The missing function!)
# ============================================================

def _parse_raw_table_rows(table_rows: List[List[str]], source_file: str, page_num: int = 1) -> List[TaskSchema]:
    """
    Parses raw table rows into TaskSchema objects using regex/heuristics.
    """
    tasks = []
    rows = [[_clean_text(c) for c in row] for row in table_rows]

    # Skip potential header
    start_idx = 0
    if rows and "id" in rows[0][0].lower():
        start_idx = 1

    for r in rows[start_idx:]:
        if not r or not any(r): continue

        # ID Check
        t_id = _parse_int_safe(r[0]) if len(r) > 0 else None
        if not t_id: continue

        # Name Check
        t_name = _clean_task_name(r[1]) if len(r) > 1 else None
        if not t_name: continue

        # Duration
        dur = _parse_duration(r[2]) if len(r) > 2 else 0

        # Dates
        start = _parse_date_flexible(r[3]) if len(r) > 3 else None
        finish = _parse_date_flexible(r[4]) if len(r) > 4 else None

        tasks.append(TaskSchema(
            task_id=t_id,
            task_name=t_name,
            duration_days=dur,
            start_date=start,
            finish_date=finish
        ))

    return tasks

def _parse_text_block_for_tasks(text: str, source_file: str, page_num: int = 1) -> List[TaskSchema]:
    """OCR fallback parser."""
    tasks = []
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        # Regex to find "123 Task Name 10d 01-01-20 01-10-20"
        m = re.match(r"^(\d+)\s+(.+?)\s+(\d+)\s*d", line, re.IGNORECASE)
        if m:
            try:
                tasks.append(TaskSchema(
                    task_id=int(m.group(1)),
                    task_name=m.group(2).strip(),
                    duration_days=int(m.group(3)),
                    start_date=None, # regex date parsing is complex, skipped for brevity
                    finish_date=None
                ))
            except:
                continue
    return tasks