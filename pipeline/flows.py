# pipeline/flows.py
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# ==========================================
# Load environment variables early
# ==========================================
project_root = Path(__file__).resolve().parent.parent
load_dotenv(project_root / ".env")

if not os.getenv("PREFECT_API_URL"):
    os.environ["PREFECT_API_URL"] = "http://127.0.0.1:4200/api"
    print("WARNING: PREFECT_API_URL not found. Defaulting to localhost:4200")

# Prefect imports AFTER env setup
from prefect import flow

from pipeline.tasks import (
    extract_document_task,
    load_to_postgres_task,
    load_to_vector_db_task,
    transform_schedule_task,
    transform_ura_task,  # <--- NEW IMPORT
)

# -------------------------------------------------------------
# FLOW: URA CIRCULAR
# -------------------------------------------------------------
@flow(name="Ingest URA Circular", log_prints=True)
def ingest_ura_flow(file_path: str):
    # 1. Extract
    data = extract_document_task(file_path, doc_type="ura_circular")

    # 2. Load SQL
    load_to_postgres_task(data, doc_type="ura_circular")

    # 3. Transform & Load Vector
    # We now properly chunk the rules instead of passing None
    chunks = transform_ura_task(data)
    load_to_vector_db_task(chunks)

# -------------------------------------------------------------
# FLOW: PROJECT SCHEDULE
# -------------------------------------------------------------
@flow(name="Ingest Schedule", log_prints=True)
def ingest_schedule_flow(file_path: str):
    # 1. Extract structured tasks
    tasks = extract_document_task(file_path, doc_type="project_schedule")

    # 2. Load structured data into SQL
    load_to_postgres_task(tasks, doc_type="project_schedule")

    # 3. Transform into row + summary semantic chunks
    chunks = transform_schedule_task(tasks)

    # 4. Load semantic embeddings into Chroma
    load_to_vector_db_task(chunks)

# -------------------------------------------------------------
# RUN DIRECTLY
# -------------------------------------------------------------
if __name__ == "__main__":
    # Define paths
    schedule_path = project_root / "data" / "Project schedule document.pdf"
    ura_path = project_root / "data" / "URA-Circular on GFA area definition.pdf"

    print("--- 🚀 Pipeline Runner ---")

    # Run Schedule Flow
    if schedule_path.exists():
        print(f"\nProcessing: {schedule_path.name}")
        ingest_schedule_flow(str(schedule_path))
    else:
        print(f"❌ File missing: {schedule_path}")

    # Run URA Flow
    if ura_path.exists():
        print(f"\nProcessing: {ura_path.name}")
        ingest_ura_flow(str(ura_path))
    else:
        print(f"❌ File missing: {ura_path}")