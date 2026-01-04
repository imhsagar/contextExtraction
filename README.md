# 🏗️ AI Context Pipeline 

> **Created by:** Himanshu Sagar

A robust ETL pipeline orchestrated with **Prefect** to extract structured and unstructured intelligence from complex construction documents (PDFs) for AI Agents.

---

### Overview
This solution transforms raw PDF documents into queryable intelligence using a **Hybrid Extraction Engine**:
1.  **Structured Layer (PostgreSQL):** Stores highly queryable entities (Gantt Tasks, Regulatory Rules) using **Django ORM**.
2.  **Semantic Layer (ChromaDB):** Stores vector embeddings of text chunks and **intelligent summaries** to enable natural language search ("Semantic Search").

###  Tech Stack
* **Orchestration:** Prefect v3 (Flows & Tasks)
* **API Framework:** Django 4.2 + Django Ninja (Async)
* **Database:** PostgreSQL 15 (Dockerized)
* **Vector Store:** ChromaDB (Local Persistence)
* **AI Logic:** Local LLM via **LM Studio** (`Qwen 2.5 7B Instruct`)
* **Extraction:** `pdfplumber`, `Camelot` (Tables), `Tesseract` (OCR).

---

### Setup & Installation
**1. Prerequisites**
   1. Docker Desktop
   2. Python 3.10
   3. LM studio (Running locally on port 1234 with Qwen 2.5 7B instruct loaded)

**2. Environment Setup**
   1. git clone https://github.com/imhsagar/contextExtraction.git
   2. cd contextExtraction
   3. ~/.pyenv/versions/3.10.14/bin/python -m venv .venv
   4. source .venv/bin/activate.fish  # (Or .bash if using bash)
   5. pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
   6. pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/rocm6.2

**3. Configuration**
    Create a .env file in the root directory. These settings are tuned for Local LLM stability

    DATABASE_URL=postgres://user:password@localhost:5432/proplens_db
    PREFECT_API_URL=http://127.0.0.1:4200/api
    ANONYMIZED_TELEMETRY=False
    EXTRACT_LLM_MODEL=qwen2.5-7b-instruct-1m
    EXTRACT_LLM_PER_CALL_TIMEOUT=120
    EXTRACT_MAX_ROWS_PER_CHUNK=5
    EXTRACT_LLM_WORKERS=2

---

### Run Instructions
1. docker-compose up -d (start postgres)
2. python manage.py migrate (create database schema)
3. prefect server start (start prefect server)
4. python -m pipeline.flows (Run the pipeline)
5. python manage.py runserver (Run server)

---

### verification
Navigate to the swagger UI: http://127.0.0.1:8000/api/docs#/
#### verify structured data:
    GET /api/tasks: Returns parsed Gantt chart rows (SQL).

    GET /api/rules: Returns extracted Regulatory Rules (SQL).

#### verify semantic search (AI):
    GET /api/search: Enter your query and execute
