import json
import zipfile
from pathlib import Path
import bm25s
import math

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

def run_bm25_for_domain(domain, query_file="lastturn"):
    base = Path("mtrag-human/retrieval_tasks") / domain

    # کورپوس
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

    queries = []
    with open(base / f"{domain}_{query_file}.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            text = q["text"].replace("|user|:", "").strip()
            queries.append((q["_id"], text))

    qrels = {}
    with open(base / "qrels" / "dev.tsv", "r", encoding="utf-8") as f:
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

    return sum(ndcg5)/len(ndcg5), sum(recall5)/len(recall5), len(ndcg5)


domains = ["clapnq", "fiqa", "govt", "cloud"]
all_ndcg, all_recall, all_n = [], [], []

print(f"{'دامنه':<10} {'nDCG@5':<10} {'Recall@5':<10} {'تعداد کوئری'}")
print("-" * 45)
for d in domains:
    ndcg, recall, n = run_bm25_for_domain(d, "lastturn")
    all_ndcg.append(ndcg * n)
    all_recall.append(recall * n)
    all_n.append(n)
    print(f"{d:<10} {ndcg:<10.4f} {recall:<10.4f} {n}")

total_n = sum(all_n)
weighted_ndcg = sum(all_ndcg) / total_n
weighted_recall = sum(all_recall) / total_n
print("-" * 45)
print(f"{'میانگین':<10} {weighted_ndcg:<10.4f} {weighted_recall:<10.4f} {total_n}")
print(f"\n(baseline رسمی مقاله: nDCG@5=0.18, Recall@5=0.20)")

print("\n" + "=" * 50)
print("مقایسه: lastturn (خام) در برابر rewrite (بازنویسی طلایی)")
print("=" * 50)
print(f"{'دامنه':<10} {'nDCG@5 (lastturn)':<20} {'nDCG@5 (rewrite)':<20} {'بهبود'}")

for d in domains:
    ndcg_last, _, _ = run_bm25_for_domain(d, "lastturn")
    ndcg_rewrite, _, _ = run_bm25_for_domain(d, "rewrite")
    improvement = ((ndcg_rewrite - ndcg_last) / ndcg_last) * 100
    print(f"{d:<10} {ndcg_last:<20.4f} {ndcg_rewrite:<20.4f} {improvement:+.1f}%")