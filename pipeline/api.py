# pipeline/api.py
import chromadb
from ninja import NinjaAPI
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

# Import your schemas and models
from pipeline.schemas import TaskSchema, RuleSchema
from pipeline.models import ProjectTask, RegulatoryRule

api = NinjaAPI(title="PropLens Context API")

# ==========================================
# SETUP VECTOR CLIENT (Must match chunk_utils)
# ==========================================
# 1. Use the SAME path as chunk_utils
CHROMA_PATH = "./chroma_db"
# 2. Use the SAME collection name as chunk_utils
COLLECTION_NAME = "project_docs"
# 3. Use the SAME model as chunk_utils
MODEL_NAME = "all-MiniLM-L6-v2"

# Load model once (global cache)
_SEARCH_MODEL = None

def get_search_model():
    global _SEARCH_MODEL
    if _SEARCH_MODEL is None:
        print("Loading Embedding Model for Search...")
        _SEARCH_MODEL = SentenceTransformer(MODEL_NAME)
    return _SEARCH_MODEL

def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    # We get the collection without defining the embedding function here,
    # because we will generate embeddings manually to match chunk_utils.
    return client.get_or_create_collection(name=COLLECTION_NAME)


# ==========================================
# API ENDPOINTS
# ==========================================

@api.get("/tasks", response=List[TaskSchema])
def list_tasks(request):
    """List all extracted tasks from SQL DB"""
    tasks = ProjectTask.objects.all()
    return tasks

# --- NEW ENDPOINT ---
@api.get("/rules", response=List[RuleSchema])
def list_rules(request):
    """List all extracted URA Regulatory Rules"""
    return RegulatoryRule.objects.all()
# --------------------

@api.get("/search")
def semantic_search(request, query: str):
    """
    Search for context using vector embeddings.
    """
    try:
        # 1. Generate embedding for the query using the SAME model
        model = get_search_model()
        query_vec = model.encode([query]).tolist()

        # 2. Search ChromaDB
        collection = get_collection()
        results = collection.query(
            query_embeddings=query_vec,
            n_results=5,
            include=["documents", "metadatas"]
        )

        # 3. Format Results
        clean_results = []
        if results and results.get('documents'):
            # Chroma returns a list of lists (for multiple queries)
            docs = results['documents'][0]
            metas = results['metadatas'][0]

            for doc, meta in zip(docs, metas):
                clean_results.append({
                    "content": doc,
                    "source": meta.get("source", "Unknown"),
                    "type": meta.get("type", "Unknown"),
                    "building": meta.get("building", "Unknown")
                })

        return {"query": query, "results": clean_results}

    except Exception as e:
        print(f"Search Error: {e}")
        return {"query": query, "results": [], "error": str(e)}