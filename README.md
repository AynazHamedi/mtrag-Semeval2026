# MTRAGEval Subtask A — Multi-Turn RAG Retrieval

A from-scratch, budget-free (CPU-only, local models) reproduction of the top-performing
approach for **SemEval-2026 Task 8 (MTRAGEval)**, Subtask A: Retrieval.

## Overview

This project builds a multi-turn RAG retrieval pipeline combining:
- **BM25** (lexical retrieval, `bm25s`)
- **Dense retrieval** with a local embedding model (`BAAI/bge-small-en-v1.5`)
- **LLM-based query rewriting** with a local model (`Qwen2.5:3b` via Ollama)
- **Reciprocal Rank Fusion (RRF)** to combine BM25 and dense results

All components run **entirely locally and free of charge** — no paid API required.

## Results

Evaluated on the official MTRAG dev set (777 answerable/partial tasks across
ClapNQ, FiQA, Govt, and Cloud domains) using the official evaluation metrics.

| Method                          | nDCG@5 (avg) |
|----------------------------------|:---:|
| BM25 + last-turn query (baseline)| 0.2395 |
| BM25 + our LLM rewrite           | 0.2823 |
| BM25 + Dense + rewrite (fusion)  | **0.3865** |
| Official paper baseline (BM25)   | 0.18 |
| Official paper baseline (Elser)  | 0.45 |

Full per-domain breakdown in [`results/`](results/).

## Architecture

```
Conversation history + question
        │
        ▼
Query Rewriting (Qwen2.5:3b, local)
        │
        ├─────────────┬
        ▼             ▼
     BM25          Dense (bge-small)
        │             │
        └──────┬──────┘
               ▼
       RRF Fusion (k=60)
               │
               ▼
         Top-10 passages
```

## Setup

### 1. Get the official MTRAG data
```bash
git clone https://github.com/IBM/mt-rag-benchmark
```
Place the `corpora/` and `mtrag-human/` folders from that repo into this project's root.

### 2. Install dependencies
```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 3. Install Ollama and pull the local model
```bash
# Download Ollama from https://ollama.com/download
ollama pull qwen2.5:3b
```

### 4. Run the pipeline
```bash
python src/build_tasks.py
python src/embed_all_domains.py
python src/rewrite_all_domains.py
python src/fusion_final.py
python src/save_final_runs.py
python src/build_predictions.py
python src/official_eval_taska.py
```

## Key findings

- Local, free open-source models (embedding + LLM rewriting) can approach or
  exceed dataset baselines relying on paid commercial APIs.
- Query rewriting benefits vary significantly by domain — FiQA (financial),
  the domain with the highest lexical mismatch, saw the largest gain (+27%).
- Combining BM25 + dense retrieval via RRF more than doubles nDCG@5 versus
  BM25 alone.

## Project context

This is part of a personal reproduction of top SemEval-2026 Task 8 systems
(AILS-NTUA, RaguTeam), built under a no-GPU, no-paid-API constraint.

## License

Code in this repository: MIT (see [LICENSE](LICENSE)).
Data: MTRAG benchmark is released by IBM under Apache 2.0
(see [mt-rag-benchmark](https://github.com/IBM/mt-rag-benchmark)).
