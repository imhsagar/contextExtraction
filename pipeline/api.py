# pipeline/api.py
import chromadb
from ninja import NinjaAPI
from typing import List
from loguru import logger
from sentence_transformers import SentenceTransformer, CrossEncoder

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

# 4. Cross-Encoder for Re-ranking (High Accuracy)
# This model is trained specifically to score how relevant a text is to a query.
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Load model once (global cache)
_SEARCH_MODEL = None
_RERANK_MODEL = None

def get_rerank_model():
    """Singleton to load the re-ranker model"""
    global _RERANK_MODEL
    if _RERANK_MODEL is None:
        logger.info("Loading Cross-Encoder for Re-ranking...")
        # Use a small, fast model to minimize latency
        _RERANK_MODEL = CrossEncoder(RERANK_MODEL_NAME)
    return _RERANK_MODEL

def get_search_model():
    global _SEARCH_MODEL
    if _SEARCH_MODEL is None:
        logger.info("Loading Embedding Model for Search...")
        _SEARCH_MODEL = SentenceTransformer(MODEL_NAME)
    return _SEARCH_MODEL

def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(name=COLLECTION_NAME)


@api.get("/tasks", response=List[TaskSchema])
def list_tasks(request):
    tasks = ProjectTask.objects.all()
    return tasks

@api.get("/rules", response=List[RuleSchema])
def list_rules(request):
    return RegulatoryRule.objects.all()
# --------------------

@api.get("/search")
def semantic_search(request, query: str):
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

        if not results or not results['documents']:
            return {"query": query, "results": []}

        retrieved_docs = results['documents'][0]
        retrieved_metas = results['metadatas'][0]

        cross_model = get_rerank_model()

        sentence_combinations = [[query, doc_text] for doc_text in retrieved_docs]

        similarity_scores = cross_model.predict(sentence_combinations)

        scored_results = []
        for idx, score in enumerate(similarity_scores):
            scored_results.append({
                "content": retrieved_docs[idx],
                "meta": retrieved_metas[idx],
                "score": float(score)  # Convert numpy float to python float
            })

        # Sort DESCENDING by the new Cross-Encoder score
        scored_results.sort(key=lambda x: x["score"], reverse=True)

        final_output = []
        for item in scored_results[:5]:  # Take only the top 5 winners
            meta = item["meta"]
            final_output.append({
                "content": item["content"],
                "score": item["score"], # Helpful for debugging relevance
                "source": meta.get("source", "Unknown"),
                "type": meta.get("type", "Unknown"),
                "building": meta.get("building", "Unknown")
            })

        return {"query": query, "results": final_output}

    except Exception as e:
        logger.error(f"Search Error: {e}")
        return {"query": query, "results": [], "error": str(e)}