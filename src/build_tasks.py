import json
from pathlib import Path

CONV_PATH = Path("mtrag-human/conversations/conversations.json")

with open(CONV_PATH, "r", encoding="utf-8") as f:
    conversations = json.load(f)

print(f"Total conversations: {len(conversations)}\n")

def get_domain(conv):
    """Extract domain from collection name"""
    name = conv["retriever"]["collection"]["name"].lower()
    for domain in ["clapnq", "fiqa", "govt", "cloud"]:
        if domain in name:
            return domain
    return "unknown"

def build_tasks_from_conversation(conv, conv_id):
    tasks = []
    history = []  
    domain = get_domain(conv)

    for msg in conv["messages"]:
        if msg["speaker"] == "user":
            task = {
                "conv_id": conv_id,
                "domain": domain,
                "history": history.copy(),   
                "question": msg["text"],
                "answerability": msg.get("enrichments", {}).get("Answerability", ["N/A"])[0],
            }
            tasks.append(task)
            history.append({"speaker": "user", "text": msg["text"]})
        else:  
            gold_passages = [c["document_id"] for c in msg.get("contexts", [])]
            if tasks:
                tasks[-1]["gold_passage_ids"] = gold_passages
                tasks[-1]["reference_answer"] = msg["text"]
            history.append({"speaker": "agent", "text": msg["text"]})

    return tasks

all_tasks = []
for i, conv in enumerate(conversations):
    all_tasks.extend(build_tasks_from_conversation(conv, conv_id=i))

print(f"Total tasks created: {len(all_tasks)}\n")

print("=" * 70)
print("Sample of 5 real tasks:")

shown = 0
for task in all_tasks:
    if len(task["history"]) >= 2 and task["answerability"] == "ANSWERABLE":
        print(f"\nTask (domain: {task['domain']})")
        print("History:")
        for h in task["history"]:
            print(f"  [{h['speaker']}] {h['text'][:100]}")
        print(f"Last question: {task['question']}")
        print(f"Answerability: {task['answerability']}")
        print(f"Number of gold passages: {len(task.get('gold_passage_ids', []))}")
        shown += 1
        if shown >= 5:
            break