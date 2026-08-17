import json
import zipfile
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DOMAINS = ["clapnq", "fiqa", "govt", "cloud"]

model = SentenceTransformer(MODEL_NAME)

for DOMAIN in DOMAINS:
    output_file = Path(f"{DOMAIN}_local_embeddings.pkl")
    if output_file.exists():
        print(f"{DOMAIN}: already done, skipping")
        continue

    print(f"{DOMAIN}: starting")

    corpus_ids = []
    corpus_texts = []

    zip_path = Path("corpora/passage_level") / f"{DOMAIN}.jsonl.zip"
    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(z.namelist()[0]) as f:
            for line in f:
                doc = json.loads(line)
                corpus_ids.append(doc["_id"])
                corpus_texts.append(doc["text"][:2000])

    print(f"{DOMAIN}: corpus size {len(corpus_ids)}")

    embeddings = model.encode(
        corpus_texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    with open(output_file, "wb") as f:
        pickle.dump({"ids": corpus_ids, "embeddings": embeddings}, f)

    print(f"{DOMAIN}: done, shape {embeddings.shape}")

print("All domains finished")