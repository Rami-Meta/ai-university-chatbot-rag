import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = "chunks.json"
INDEX_FILE = "faiss.index"
METADATA_FILE = "metadata.json"


def main():
    print("Loading chunks...")
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [c["text"] for c in chunks]

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Creating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)

    embeddings = np.array(embeddings).astype("float32")

    print("Building FAISS index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    print("Saving index...")
    faiss.write_index(index, INDEX_FILE)

    print("Saving metadata...")
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print("Done! Index built successfully.")


if __name__ == "__main__":
    main()
