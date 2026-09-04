"""
RAG Agent — answers questions from your own documents using OpenAI's API.

How it works (the whole pipeline, in plain terms):
1. Load documents (.txt and .pdf) from the docs/ folder.
2. Split each document into small overlapping chunks (so the model gets
   focused context instead of a whole file at once).
3. Convert every chunk into an embedding (a vector of numbers that
   captures its meaning) using OpenAI's embedding model.
4. When the user asks a question, embed the question the same way,
   and find the chunks whose embeddings are most similar (cosine
   similarity) — this is the "retrieval" step.
5. Stuff those top chunks into a prompt and ask the chat model to
   answer using ONLY that context — this is the "generation" step.

This is intentionally dependency-light (no vector database, no
framework) so every step is visible and explainable in an interview.
"""

import os
import glob
import numpy as np
from openai import OpenAI
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()  # reads OPENAI_API_KEY from a local .env file

client = OpenAI()  # picks up OPENAI_API_KEY from the environment

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 150    # overlap so we don't cut sentences in half
TOP_K = 3              # how many chunks to retrieve per question


def load_documents(folder="docs"):
    """Read every .txt and .pdf file in `folder` and return raw text per file."""
    texts = {}
    for path in glob.glob(os.path.join(folder, "*")):
        if path.lower().endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                texts[path] = f.read()
        elif path.lower().endswith(".pdf"):
            reader = PdfReader(path)
            texts[path] = "\n".join(page.extract_text() or "" for page in reader.pages)
    return texts


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks so context isn't lost at boundaries."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return [c.strip() for c in chunks if c.strip()]


def embed(texts):
    """Get embeddings for a list of strings in one API call."""
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def build_index(folder="docs"):
    """Load, chunk, and embed every document. Returns (chunks, vectors, sources)."""
    documents = load_documents(folder)
    all_chunks, all_sources = [], []
    for path, text in documents.items():
        for chunk in chunk_text(text):
            all_chunks.append(chunk)
            all_sources.append(os.path.basename(path))

    if not all_chunks:
        raise ValueError(f"No .txt or .pdf files found in '{folder}/'. Add some documents first.")

    print(f"Embedding {len(all_chunks)} chunks from {len(documents)} document(s)...")
    vectors = embed(all_chunks)
    return all_chunks, vectors, all_sources


def retrieve(question, chunks, vectors, sources, top_k=TOP_K):
    """Find the top_k chunks most relevant to the question."""
    q_vector = embed([question])[0]
    scores = [cosine_similarity(q_vector, v) for v in vectors]
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top_indices = ranked[:top_k]
    return [(chunks[i], sources[i], scores[i]) for i in top_indices]


def answer_question(question, chunks, vectors, sources):
    """Retrieve relevant context, then ask the chat model to answer from it."""
    results = retrieve(question, chunks, vectors, sources)
    context = "\n\n---\n\n".join(f"[Source: {src}]\n{chunk}" for chunk, src, _ in results)

    prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't have enough information in the documents to answer that."

Context:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content, results


def main():
    print("=== RAG Agent (answers from your documents) ===")
    chunks, vectors, sources = build_index("docs")
    print(f"Index ready. Loaded {len(chunks)} chunks. Ask questions below (type 'exit' to quit).\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        answer, sources_used = answer_question(question, chunks, vectors, sources)
        print(f"\nAgent: {answer}\n")
        print("Sources used:")
        for chunk, src, score in sources_used:
            preview = chunk[:80].replace("\n", " ")
            print(f"  - {src} (similarity: {score:.2f}): \"{preview}...\"")
        print()


if __name__ == "__main__":
    main()
