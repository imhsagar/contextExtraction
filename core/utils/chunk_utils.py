# core/utils/chunk_utils.py

import uuid
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

from pipeline.schemas import TaskSchema


_MODEL = None
_CHROMA = None

def _get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def _get_chroma(persist_dir="./chroma_db"):
    global _CHROMA
    if _CHROMA is None:
        _CHROMA = PersistentClient(path=persist_dir)
    return _CHROMA

def aggregate_tasks_by_building(tasks: List[TaskSchema]):
    groups = {}

    for t in tasks:
        building = getattr(t, "building", None) or "UNSPECIFIED"
        groups.setdefault(building, []).append(t)

    summaries = {}

    for building, rows in groups.items():
        total_days = sum((r.duration_days or 0) for r in rows)
        longest = max(rows, key=lambda r: (r.duration_days or 0))

        summaries[building] = {
            "building": building,
            "tasks": rows,
            "num_tasks": len(rows),
            "total_duration_days": total_days,
            "longest_task": {
                "task_id": longest.task_id,
                "task_name": longest.task_name,
                "duration_days": longest.duration_days,
            }
        }

    return summaries

def create_semantic_chunks(tasks: List[TaskSchema], summaries: Dict[str, Dict]):
    row_chunks = []
    summary_chunks = []

    # Row-Level Chunks
    for t in tasks:
        text = (
            f"Task {t.task_id}: {t.task_name}. "
            f"Duration: {t.duration_days} days. "
        )
        if t.start_date:
            text += f"Start: {t.start_date}. "
        if t.finish_date:
            text += f"Finish: {t.finish_date}. "

        building = getattr(t, "building", None) or "UNKNOWN"

        meta = {
            "type": "task",
            "building": building,
            "task_id": t.task_id,
            "source": "Schedule",
        }

        row_chunks.append({
            "id": str(uuid.uuid4()),
            "text": text.strip(),
            "metadata": meta,
        })

    # Summary-Level Chunks
    for b, s in summaries.items():
        lines = [
            f"{b} — Summary:",
            f"Total tasks: {s['num_tasks']}",
            f"Total duration: {s['total_duration_days']} days",
            f"Longest task: {s['longest_task']['task_name']} ({s['longest_task']['duration_days']} days)",
        ]

        for t in sorted(s["tasks"], key=lambda r: r.duration_days or 0, reverse=True):
            lines.append(f"- {t.task_name} ({t.duration_days} days)")

        text = "\n".join(lines)

        summary_chunks.append({
            "id": str(uuid.uuid4()),
            "text": text,
            "metadata": {"type": "summary", "building": b, "source": "Schedule"},
        })

    return row_chunks, summary_chunks

# INDEX INTO CHROMA (NEW API)
def _sanitize_metadata(meta: dict) -> dict:
    """
    Chroma metadata cannot contain None values.
    Replace None with 'UNKNOWN'.
    """
    clean = {}
    for k, v in meta.items():
        if v is None:
            clean[k] = "UNKNOWN"
        else:
            clean[k] = v
    return clean


def index_chunks_to_chroma(row_chunks: List[Dict], summary_chunks: List[Dict], persist_dir="./chroma_db") -> int:
    client = _get_chroma(persist_dir)
    model = _get_model()

    collection = client.get_or_create_collection(
        name="project_docs",
        metadata={"hnsw:space": "cosine"},
    )

    all_chunks = row_chunks + summary_chunks

    documents = [c["text"] for c in all_chunks]
    metadatas = [_sanitize_metadata(c["metadata"]) for c in all_chunks]
    ids = [c["id"] for c in all_chunks]
    embeddings = model.encode(documents).tolist()

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings,
    )

    return len(all_chunks)

