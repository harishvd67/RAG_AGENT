# RAG Agent — Document Q&A with OpenAI

A small AI agent that answers questions using only the content of documents
you give it (Retrieval-Augmented Generation). Ask it something, and it
retrieves the most relevant chunks from your files, then uses an LLM to
answer based only on that retrieved context — not from the model's general
training knowledge.

## How it works

1. **Load** — reads every `.txt` and `.pdf` file in `docs/`.
2. **Chunk** — splits documents into overlapping ~800-character chunks.
3. **Embed** — converts each chunk into a vector using OpenAI's
   `text-embedding-3-small` model.
4. **Retrieve** — when you ask a question, it's embedded the same way, and
   the most similar chunks are found using cosine similarity.
5. **Generate** — the top chunks are inserted into a prompt, and
   `gpt-4o-mini` answers strictly from that context.

No vector database or agent framework is used on purpose — every step is
plain Python so it's easy to explain and extend.

## Setup

```bash
# 1. Clone and enter the project
git clone <your-repo-url>
cd rag-agent

# 2. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your OpenAI API key
cp .env.example .env
# then open .env and paste your key from https://platform.openai.com/api-keys
```

## Usage

Drop your own `.txt` or `.pdf` files into the `docs/` folder (a sample file
is included so it works out of the box), then run:

```bash
python rag_agent.py
```

Example session:

```
You: What internships has this person done?
Agent: They completed a Python Internship at Slash Mark IT Solutions,
building a real-time face recognition system, and an AI Internship
at SkillDzire, Hyderabad, training AI models for predictive tasks.

Sources used:
  - sample_about_me.txt (similarity: 0.81): "V Harish is an MCA graduate..."
```

Type `exit` to quit.

## Possible extensions

- Swap the in-memory similarity search for a real vector database (FAISS,
  Chroma, Pinecone) for larger document sets.
- Add a simple web UI (Streamlit/Flask) instead of the command line.
- Add a second "tool" the agent can call — e.g., a calculator or web search
  — to move from pure retrieval toward a true multi-tool agent.

## Tech stack

Python, OpenAI API (chat + embeddings), pypdf, NumPy
