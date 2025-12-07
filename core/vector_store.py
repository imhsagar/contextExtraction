import chromadb
from chromadb.utils import embedding_functions

class VectorManager:
    def __init__(self, collection_name="proplens_docs"):
        # Persist data to disk so it survives restarts
        self.client = chromadb.PersistentClient(path="./chroma_db")

        # Use default embedding function (Sentence Transformers)
        self.emb_fn = embedding_functions.DefaultEmbeddingFunction()

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.emb_fn
        )

    def add_document(self, doc_id: str, text: str, metadata: dict):
        """Add a text chunk to the vector DB"""
        self.collection.add(
            ids=[doc_id],
            documents=[text],
            metadatas=[metadata]
        )

    def search(self, query: str, n_results=3):
        """Semantic search for AI Agents"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results