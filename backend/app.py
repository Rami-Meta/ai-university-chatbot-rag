import json
import faiss
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from openai import OpenAI

INDEX_FILE = "faiss.index"
METADATA_FILE = "metadata.json"

app = FastAPI()
client = OpenAI()

print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading FAISS index...")
index = faiss.read_index(INDEX_FILE)

print("Loading metadata...")
with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)


class QueryRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "RAG chatbot API is running"}


@app.post("/ask")
def ask_question(request: QueryRequest):
    question = request.question

    query_embedding = model.encode([question])
    query_embedding = np.array(query_embedding).astype("float32")

    k = 5
    distances, indices = index.search(query_embedding, k)

    retrieved_chunks = []
    for idx in indices[0]:
        chunk = metadata[idx]
        retrieved_chunks.append(chunk)

    source_counts = {}
    for chunk in retrieved_chunks:
        source = chunk["source"]
        source_counts[source] = source_counts.get(source, 0) + 1

    best_source = max(source_counts, key=source_counts.get)

    filtered_chunks = [c for c in retrieved_chunks if c["source"] == best_source][:3]

    results = []
    context_texts = []

    for chunk in filtered_chunks:
        results.append(
            {"source": chunk["source"], "page": chunk["page"], "text": chunk["text"]}
        )
        context_texts.append(
            f"Source: {chunk['source']}, Page: {chunk['page']}\n{chunk['text']}"
        )

    context = "\n\n".join(context_texts)

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=f"""
You are a helpful university assistant.

Answer the question clearly using only the context below.
If the answer is not in the context, say you could not find it in the documents.
Summarize the information in simple sentences.

Question:
{question}

Context:
{context}
""",
        )
        answer = response.output_text
    except Exception as e:
        answer = f"OpenAI API error: {str(e)}"

    return {"question": question, "answer": answer, "sources": results}
