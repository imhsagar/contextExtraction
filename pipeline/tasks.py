# pipeline/tasks.py
import os
import sys
import django
import uuid
from pathlib import Path
from prefect import task

# ==========================================
# 1. BOOTSTRAP DJANGO
# ==========================================
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "proplens.settings")

try:
    django.setup()
except RuntimeError:
    pass

from core.extractor import DocumentExtractor
from pipeline.models import RegulatoryRule, ProjectTask
from core.utils.chunk_utils import index_chunks_to_chroma, aggregate_tasks_by_building, create_semantic_chunks

# -------------------------------------------------------------
# EXTRACT
# -------------------------------------------------------------
@task(name="Extract Data", log_prints=True)
def extract_document_task(file_path: str, doc_type: str):
    extractor = DocumentExtractor()
    print(f"Starting extraction for: {file_path}")

    if doc_type == "ura_circular":
        extractor.extract_images_from_ura(file_path)
        return extractor.extract_ura_rules_with_llm(file_path)

    elif doc_type == "project_schedule":
        return extractor.extract_project_schedule_hybrid(file_path)

    return []

# -------------------------------------------------------------
# TRANSFORM
# -------------------------------------------------------------
@task(name="Transform Schedule Data", log_prints=True)
def transform_schedule_task(data):
    """Build: summaries + row chunks for Schedules."""
    if not data:
        print("No schedule data to transform.")
        return None

    summaries = aggregate_tasks_by_building(data)
    row_chunks, summary_chunks = create_semantic_chunks(data, summaries)

    print(f"Created {len(row_chunks)} row chunks and {len(summary_chunks)} summary chunks.")
    return (row_chunks, summary_chunks)

@task(name="Transform URA Data", log_prints=True)
def transform_ura_task(data):
    """
    Convert RegulatoryRule objects into semantic chunks for Vector DB.
    """
    if not data:
        return []

    chunks = []
    for rule in data:
        # Create a rich text representation for the AI
        text = f"Rule {rule.rule_id}: {rule.rule_summary}. Measurement Basis: {rule.measurement_basis}"

        chunks.append({
            "id": str(uuid.uuid4()),
            "text": text,
            "metadata": {
                "source": "URA-Circular",
                "type": "rule",
                "rule_id": str(rule.rule_id)
            }
        })

    print(f"Created {len(chunks)} semantic chunks for URA rules.")
    return chunks

# -------------------------------------------------------------
# LOAD: POSTGRES
# -------------------------------------------------------------
@task(name="Load Postgres", log_prints=True)
def load_to_postgres_task(data, doc_type: str):
    if not data:
        print("No data to save to PostgreSQL.")
        return

    print(f"Saving {len(data)} SQL records for: {doc_type}")

    if doc_type == "ura_circular":
        objs = [
            RegulatoryRule(
                rule_id=item.rule_id,
                rule_summary=item.rule_summary,
                measurement_basis=item.measurement_basis
            )
            for item in data
        ]
        RegulatoryRule.objects.bulk_create(objs, ignore_conflicts=True)

    elif doc_type == "project_schedule":
        objs = [
            ProjectTask(
                task_id=item.task_id,
                task_name=item.task_name,
                duration_days=item.duration_days,
                start_date=item.start_date,
                finish_date=item.finish_date,
            )
            for item in data
        ]
        ProjectTask.objects.bulk_create(objs, ignore_conflicts=True)

# -------------------------------------------------------------
# LOAD: VECTOR DATABASE (Chroma)
# -------------------------------------------------------------
@task(name="Load Vector DB", log_prints=True)
def load_to_vector_db_task(chunks):
    if not chunks:
        print("No chunks to save to Vector DB.")
        return

    row_chunks = []
    summary_chunks = []

    # Detect input type
    if isinstance(chunks, tuple):
        row_chunks, summary_chunks = chunks
    elif isinstance(chunks, list):
        row_chunks = chunks

    total = index_chunks_to_chroma(row_chunks, summary_chunks)
    print(f"Indexed {total} total semantic chunks into ChromaDB.")