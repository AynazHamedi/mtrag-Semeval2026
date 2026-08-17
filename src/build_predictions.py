import json
from pathlib import Path

base_path = Path("mtrag-human/generation_tasks/RAG.jsonl")
base_tasks = {}
with open(base_path, "r", encoding="utf-8") as f:
    for line in f:
        task = json.loads(line)
        base_tasks[task["task_id"]] = task

DOMAINS = ["clapnq", "fiqa", "govt", "cloud"]
all_predictions = []

for domain in DOMAINS:
    run_path = Path("subtask_a_runs") / f"{domain}_run.tsv"
    contexts_by_qid = {}
    with open(run_path, "r", encoding="utf-8") as f:
        for line in f:
            qid, q0, doc_id, rank, score, tag = line.strip().split("\t")
            contexts_by_qid.setdefault(qid, []).append({
                "document_id": doc_id,
                "score": float(score),
            })

    for qid, contexts in contexts_by_qid.items():
        if qid not in base_tasks:
            print(f"Warning: task_id {qid} not found in base file")
            continue
        base_task = base_tasks[qid]
        prediction = {
            "conversation_id": base_task["conversation_id"],
            "task_id": base_task["task_id"],
            "Collection": base_task["Collection"],
            "input": base_task["input"],
            "contexts": contexts,
        }
        all_predictions.append(prediction)

output_path = Path("subtask_a_predictions.jsonl")
with open(output_path, "w", encoding="utf-8") as f:
    for pred in all_predictions:
        f.write(json.dumps(pred) + "\n")

print(f"Total predictions written: {len(all_predictions)}")
print(f"Saved to {output_path}")