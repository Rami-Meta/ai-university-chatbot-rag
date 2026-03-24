import os
import json
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from openai import OpenAI

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE = os.path.join(BASE_DIR, "faiss.index")
METADATA_FILE = os.path.join(BASE_DIR, "metadata.json")

app = FastAPI()
client = OpenAI()

model = None
index = None
metadata = None


class QueryRequest(BaseModel):
    question: str


@app.on_event("startup")
def load_resources():
    global model, index, metadata

    print("Loading model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Loading FAISS index...")
    index = faiss.read_index(INDEX_FILE)

    print("Loading metadata...")
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print("Startup complete.")


@app.get("/")
def home():
    return {"message": "RAG chatbot API is running"}


@app.post("/ask")
def ask_question(request: QueryRequest):
    global model, index, metadata

    if model is None or index is None or metadata is None:
        raise HTTPException(
            status_code=503,
            detail="Model or index still loading. Try again in a moment.",
        )

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    query_embedding = model.encode([question])
    query_embedding = np.array(query_embedding).astype("float32")

    k = 3
    distances, indices = index.search(query_embedding, k)

    retrieved_chunks = []
    for i in indices[0]:
        if 0 <= i < len(metadata):
            chunk = metadata[i]
            if isinstance(chunk, dict) and "text" in chunk:
                retrieved_chunks.append(chunk["text"])
            else:
                retrieved_chunks.append(str(chunk))

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are a helpful university assistant chatbot.
Answer the question using only the context below.
If the answer is not in the context, say: "I could not find that in the university data."

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful university assistant chatbot.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    answer = response.choices[0].message.content

    return {"question": question, "answer": answer, "context": retrieved_chunks}
