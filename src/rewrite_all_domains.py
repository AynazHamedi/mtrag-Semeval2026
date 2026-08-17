import json
import ollama
from pathlib import Path
from tqdm import tqdm

DOMAINS = ["clapnq", "fiqa", "govt", "cloud"]

CONV_PATH = Path("mtrag-human/conversations/conversations.json")
with open(CONV_PATH, "r", encoding="utf-8") as f:
    conversations = json.load(f)

def get_domain(conv):
    name = conv["domain"].lower()
    for d in DOMAINS:
        if d in name:
            return d
    return "unknown"

def build_tasks_from_conversation(conv, conv_id):
    tasks = []
    history = []
    domain = get_domain(conv)
    user_texts_so_far = []
    for msg in conv["messages"]:
        if msg["speaker"] == "user":
            user_texts_so_far.append(msg["text"])
            key = "\n".join(f"|user|: {t}" for t in user_texts_so_far)
            task = {
                "conv_id": conv_id,
                "domain": domain,
                "history": history.copy(),
                "question": msg["text"],
                "match_key": key,
            }
            tasks.append(task)
            history.append({"speaker": "user", "text": msg["text"]})
        else:
            history.append({"speaker": "agent", "text": msg["text"]})
    return tasks

all_tasks = []
for i, conv in enumerate(conversations):
    all_tasks.extend(build_tasks_from_conversation(conv, conv_id=i))

lookup = {}
for task in all_tasks:
    lookup[(task["domain"], task["match_key"])] = task

def rewrite_question(history, question):
    if not history:
        return question
    history_text = "\n".join(f"{h['speaker']}: {h['text']}" for h in history)
    prompt = f"""Conversation history:
{history_text}

Current question: {question}

Rewrite the current question so it is fully standalone and understandable without the conversation history. Keep it as close as possible to the original wording. Output only the rewritten question, nothing else."""
    response = ollama.generate(model="qwen2.5:3b", prompt=prompt)
    return response["response"].strip()

for domain in DOMAINS:
    input_path = Path("mtrag-human/retrieval_tasks") / domain / f"{domain}_questions.jsonl"
    output_path = Path(f"{domain}_our_rewrite.jsonl")

    if output_path.exists():
        print(f"{domain}: output already exists, skipping")
        continue

    queries = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line))

    print(f"{domain}: processing {len(queries)} queries")

    results = []
    not_found = 0

    for q in tqdm(queries):
        key = (domain, q["text"])
        task = lookup.get(key)
        if task is None:
            not_found += 1
            results.append({"_id": q["_id"], "text": q["text"]})
            continue
        rewritten = rewrite_question(task["history"], task["question"])
        results.append({"_id": q["_id"], "text": f"|user|: {rewritten}"})

    print(f"{domain}: not_found = {not_found}")

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"{domain}: saved to {output_path}")

print("All domains finished")