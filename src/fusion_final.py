import json
import zipfile
import pickle
import math
from pathlib import Path
import bm25s
from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "BAAI/bge-small-en-v1.5"
model = SentenceTransformer(MODEL_NAME)

def dcg_at_k(relevances, k):
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))

def ndcg_at_k(ranked_ids, relevant_set, k):
    relevances = [1 if cid in relevant_set else 0 for cid in ranked_ids]
    dcg = dcg_at_k(relevances, k)
    idcg = dcg_at_k(sorted(relevances, reverse=True), k)
    return dcg / idcg if idcg > 0 else 0.0

def recall_at_k(ranked_ids, relevant_set, k):
    if not relevant_set:
        return 0.0
    hits = sum(1 for cid in ranked_ids[:k] if cid in relevant_set)
    return hits / len(relevant_set)

def load_queries(path):
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            text = q["text"].replace("|user|:", "").strip()
            queries.append((q["_id"], text))
    return queries

def rrf_fuse(rank_lists, k=60):
    scores = {}
    for ranked_ids in rank_lists:
        for rank, doc_id in enumerate(ranked_ids):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: -x[1])]

def run_fusion(domain):
    corpus_ids, corpus_texts = [], []
    zip_path = Path("corpora/passage_level") / f"{domain}.jsonl.zip"
    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(z.namelist()[0]) as f:
            for line in f:
                doc = json.loads(line)
                corpus_ids.append(doc["_id"])
                corpus_texts.append(doc["text"])

    corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en")
    bm25_retriever = bm25s.BM25()
    bm25_retriever.index(corpus_tokens)

    with open(f"{domain}_local_embeddings.pkl", "rb") as f:
        dense_data = pickle.load(f)
    dense_ids = dense_data["ids"]
    dense_embeddings = dense_data["embeddings"]

    qrels = {}
    with open(Path("mtrag-human/retrieval_tasks") / domain / "qrels" / "dev.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            qid, cid, score = line.strip().split("\t")
            qrels.setdefault(qid, set()).add(cid)

    queries = load_queries(Path(f"{domain}_our_rewrite.jsonl"))
    query_texts = [q[1] for q in queries]

    query_tokens = bm25s.tokenize(query_texts, stopwords="en")
    bm25_results, _ = bm25_retriever.retrieve(query_tokens, k=20)

    query_embeddings = model.encode(query_texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    similarity_matrix = query_embeddings @ dense_embeddings.T

    ndcg5, recall5 = [], []

    for i, (qid, _) in enumerate(queries):
        if qid not in qrels:
            continue

        bm25_ranked = [corpus_ids[idx] for idx in bm25_results[i]]
        dense_top_idx = np.argsort(-similarity_matrix[i])[:20]
        dense_ranked = [dense_ids[idx] for idx in dense_top_idx]

        fused = rrf_fuse([bm25_ranked, dense_ranked])[:10]

        ndcg5.append(ndcg_at_k(fused, qrels[qid], 5))
        recall5.append(recall_at_k(fused, qrels[qid], 5))

    return sum(ndcg5) / len(ndcg5), sum(recall5) / len(recall5)

DOMAINS = ["clapnq", "fiqa", "govt", "cloud"]

print(f"{'domain':<10} {'ndcg5':<10} {'recall5':<10}")
print("-" * 30)

all_ndcg, all_recall = [], []
for domain in DOMAINS:
    ndcg, recall = run_fusion(domain)
    all_ndcg.append(ndcg)
    all_recall.append(recall)
    print(f"{domain:<10} {ndcg:<10.4f} {recall:<10.4f}")

print(f"{'average':<10} {sum(all_ndcg)/4:<10.4f} {sum(all_recall)/4:<10.4f}")
print("Comparison to previous results:")
print("BM25 lastturn average ndcg5: 0.2395")
print("BM25 our_rewrite average ndcg5: 0.2823")
print("Dense fiqa only ndcg5: 0.3329")