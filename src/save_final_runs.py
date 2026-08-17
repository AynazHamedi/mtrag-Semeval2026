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

def run_fusion_and_save(domain, output_dir):
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

    queries = load_queries(Path(f"{domain}_our_rewrite.jsonl"))
    query_texts = [q[1] for q in queries]

    query_tokens = bm25s.tokenize(query_texts, stopwords="en")
    bm25_results, _ = bm25_retriever.retrieve(query_tokens, k=20)

    query_embeddings = model.encode(query_texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    similarity_matrix = query_embeddings @ dense_embeddings.T

    output_path = output_dir / f"{domain}_run.tsv"
    with open(output_path, "w", encoding="utf-8") as f:
        for i, (qid, _) in enumerate(queries):
            bm25_ranked = [corpus_ids[idx] for idx in bm25_results[i]]
            dense_top_idx = np.argsort(-similarity_matrix[i])[:20]
            dense_ranked = [dense_ids[idx] for idx in dense_top_idx]

            fused = rrf_fuse([bm25_ranked, dense_ranked])[:10]

            for rank, doc_id in enumerate(fused):
                score = 1.0 / (rank + 1)
                f.write(f"{qid}\tQ0\t{doc_id}\t{rank+1}\t{score:.6f}\trun1\n")

    print(f"{domain}: saved run file to {output_path}")

DOMAINS = ["clapnq", "fiqa", "govt", "cloud"]
output_dir = Path("subtask_a_runs")
output_dir.mkdir(exist_ok=True)

for domain in DOMAINS:
    run_fusion_and_save(domain, output_dir)

print("All run files saved")