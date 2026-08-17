import json
import zipfile
import math
from pathlib import Path
import bm25s

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

def run_bm25(domain, queries):
    corpus_ids, corpus_texts = [], []
    zip_path = Path("corpora/passage_level") / f"{domain}.jsonl.zip"
    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(z.namelist()[0]) as f:
            for line in f:
                doc = json.loads(line)
                corpus_ids.append(doc["_id"])
                corpus_texts.append(doc["text"])

    corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en")
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    qrels = {}
    with open(Path("mtrag-human/retrieval_tasks") / domain / "qrels" / "dev.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            qid, cid, score = line.strip().split("\t")
            qrels.setdefault(qid, set()).add(cid)

    query_texts = [q[1] for q in queries]
    query_tokens = bm25s.tokenize(query_texts, stopwords="en")
    results, scores = retriever.retrieve(query_tokens, k=10)

    ndcg5, recall5 = [], []
    for i, (qid, _) in enumerate(queries):
        if qid not in qrels:
            continue
        ranked_ids = [corpus_ids[idx] for idx in results[i]]
        ndcg5.append(ndcg_at_k(ranked_ids, qrels[qid], 5))
        recall5.append(recall_at_k(ranked_ids, qrels[qid], 5))

    return sum(ndcg5) / len(ndcg5), sum(recall5) / len(recall5)

DOMAINS = ["clapnq", "fiqa", "govt", "cloud"]

print(f"{'domain':<10} {'lastturn':<12} {'our_rewrite':<12} {'gold_rewrite':<12}")

all_last, all_ours, all_gold = [], [], []

for domain in DOMAINS:
    base = Path("mtrag-human/retrieval_tasks") / domain

    q_last = load_queries(base / f"{domain}_lastturn.jsonl")
    q_ours = load_queries(Path(f"{domain}_our_rewrite.jsonl"))
    q_gold = load_queries(base / f"{domain}_rewrite.jsonl")

    ndcg_last, _ = run_bm25(domain, q_last)
    ndcg_ours, _ = run_bm25(domain, q_ours)
    ndcg_gold, _ = run_bm25(domain, q_gold)

    all_last.append(ndcg_last)
    all_ours.append(ndcg_ours)
    all_gold.append(ndcg_gold)

    print(f"{domain:<10} {ndcg_last:<12.4f} {ndcg_ours:<12.4f} {ndcg_gold:<12.4f}")

print("-" * 50)
print(f"{'average':<10} {sum(all_last)/4:<12.4f} {sum(all_ours)/4:<12.4f} {sum(all_gold)/4:<12.4f}")