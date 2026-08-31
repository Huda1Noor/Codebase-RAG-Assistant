import json
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi


class HybridRetriever:
    def __init__(self, chunks_path="data/chunks.json", chroma_path="data/chroma_db", collection_name="flask_codebase"):
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        self.documents = [c["code"] for c in self.chunks]
        self.metadatas = [
            {
                "file": c["file"],
                "name": c["name"],
                "type": c["type"],
                "start_line": c["start_line"],
                "end_line": c["end_line"],
            }
            for c in self.chunks
        ]

        # Dense retrieval setup
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)

        # Sparse retrieval setup
        tokenized_corpus = [doc.split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Re-ranker
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def dense_search(self, query, n=10):
        query_embedding = self.embed_model.encode(query).tolist()
        results = self.collection.query(query_embeddings=[query_embedding], n_results=n)
        return results["ids"][0]

    def sparse_search(self, query, n=10):
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:n]
        return [f"chunk_{idx}" for idx in top_indices]

    def fuse(self, dense_ids, sparse_ids, k=60):
        scores = {}
        for rank, doc_id in enumerate(dense_ids):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        for rank, doc_id in enumerate(sparse_ids):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        return sorted(scores.items(), key=lambda x: -x[1])

    def rerank(self, query, candidate_ids, top_k=5):
        candidate_texts = [self.documents[int(cid.split("_")[1])] for cid in candidate_ids]
        pairs = [[query, text] for text in candidate_texts]
        rerank_scores = self.reranker.predict(pairs)
        reranked = sorted(zip(candidate_ids, rerank_scores), key=lambda x: -x[1])
        return reranked[:top_k]

    def retrieve(self, query, n_candidates=10, top_k=5):
        """Full pipeline: dense + sparse -> fuse -> rerank -> top_k results with metadata."""
        dense_ids = self.dense_search(query, n=n_candidates)
        sparse_ids = self.sparse_search(query, n=n_candidates)
        fused = self.fuse(dense_ids, sparse_ids)
        fused_ids = [doc_id for doc_id, _ in fused[:n_candidates]]
        reranked = self.rerank(query, fused_ids, top_k=top_k)

        final_results = []
        for doc_id, score in reranked:
            idx = int(doc_id.split("_")[1])
            final_results.append({
                "score": float(score),
                "code": self.documents[idx],
                **self.metadatas[idx]
            })
        return final_results


if __name__ == "__main__":
    retriever = HybridRetriever()
    results = retriever.retrieve("how does flask handle url routing")
    for r in results:
        print(f"{r['score']:.4f} | {r['file']} :: {r['name']} (lines {r['start_line']}-{r['end_line']})")