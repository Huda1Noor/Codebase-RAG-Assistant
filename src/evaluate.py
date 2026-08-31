import json
from retriever import HybridRetriever


def load_eval_set(path="data/eval_set.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_at_k(retrieved_names, correct_names, k):
    top_k = retrieved_names[:k]
    hits = [name for name in correct_names if name in top_k]
    return 1.0 if hits else 0.0


def mrr(retrieved_names, correct_names):
    for rank, name in enumerate(retrieved_names, start=1):
        if name in correct_names:
            return 1.0 / rank
    return 0.0


def evaluate_method(retriever, eval_set, method, k_values=(3, 5, 10)):
    """method: one of 'dense', 'sparse', 'hybrid', 'hybrid_rerank'"""
    recalls = {k: [] for k in k_values}
    mrrs = []

    for item in eval_set:
        query = item["question"]
        correct_names = item["correct_chunks"]

        if method == "dense":
            ids = retriever.dense_search(query, n=10)
        elif method == "sparse":
            ids = retriever.sparse_search(query, n=10)
        elif method == "hybrid":
            dense_ids = retriever.dense_search(query, n=10)
            sparse_ids = retriever.sparse_search(query, n=10)
            fused = retriever.fuse(dense_ids, sparse_ids,k=5)
            ids = [doc_id for doc_id, _ in fused[:10]]
        elif method == "hybrid_rerank":
            dense_ids = retriever.dense_search(query, n=10)
            sparse_ids = retriever.sparse_search(query, n=10)
            fused = retriever.fuse(dense_ids, sparse_ids, k=5)
            fused_ids = [doc_id for doc_id, _ in fused[:10]]
            reranked = retriever.rerank(query, fused_ids, top_k=10)
            ids = [doc_id for doc_id, _ in reranked]
        else:
            raise ValueError(f"Unknown method: {method}")

        # Convert chunk IDs to function names for comparison
        retrieved_names = [retriever.metadatas[int(cid.split("_")[1])]["name"] for cid in ids]

        for k in k_values:
            recalls[k].append(recall_at_k(retrieved_names, correct_names, k))
        mrrs.append(mrr(retrieved_names, correct_names))

    results = {f"recall@{k}": sum(v) / len(v) for k, v in recalls.items()}
    results["mrr"] = sum(mrrs) / len(mrrs)
    return results


if __name__ == "__main__":
    retriever = HybridRetriever()
    eval_set = load_eval_set()

    methods = ["dense", "sparse", "hybrid", "hybrid_rerank"]
    print(f"{'Method':<15} | Recall@3 | Recall@5 | Recall@10 | MRR")
    print("-" * 65)

    for method in methods:
        results = evaluate_method(retriever, eval_set, method)
        print(f"{method:<15} | {results['recall@3']:.2f}     | {results['recall@5']:.2f}     | {results['recall@10']:.2f}      | {results['mrr']:.2f}")