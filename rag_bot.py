"""
RAG QA Bot — Retrieval Augmented Generation Question Answering System
Author: Tharuna Rajpurohit
GitHub: github.com/Era-Of-Tharuna

How it works:
1. Load documents (PDF or text files)
2. Split them into small chunks
3. Convert each chunk into an embedding (vector) using OpenAI
4. Store embeddings in Pinecone vector database
5. When user asks a question, convert question to embedding
6. Find the most similar chunks in Pinecone
7. Send those chunks + question to GPT to generate a final answer
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# ── Clients ──────────────────────────────────────────────────────────────────

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

INDEX_NAME = "rag-qa-bot"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-3.5-turbo"
CHUNK_SIZE = 500      # characters per chunk
CHUNK_OVERLAP = 50    # overlap between chunks to preserve context
TOP_K = 3             # number of similar chunks to retrieve


# ── Step 1: Chunking ─────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split a long text into overlapping chunks.
    Overlap helps preserve context at chunk boundaries.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def load_documents(file_paths: list[str]) -> list[dict]:
    """
    Load text from .txt or .pdf files.
    Returns a list of {filename, text} dicts.
    """
    documents = []
    for path in file_paths:
        if path.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        elif path.endswith(".pdf"):
            try:
                import PyPDF2
                with open(path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    text = " ".join(page.extract_text() for page in reader.pages)
            except ImportError:
                print("PyPDF2 not installed. Run: pip install PyPDF2")
                continue
        else:
            print(f"Unsupported file type: {path}")
            continue
        documents.append({"filename": os.path.basename(path), "text": text})
    return documents


# ── Step 2: Embeddings ────────────────────────────────────────────────────────

def get_embedding(text: str) -> list[float]:
    """
    Convert text to a vector embedding using OpenAI.
    Each embedding is a list of 1536 floats that captures semantic meaning.
    """
    response = openai_client.embeddings.create(
        input=text,
        model=EMBED_MODEL
    )
    return response.data[0].embedding


# ── Step 3: Pinecone Setup ────────────────────────────────────────────────────

def setup_index():
    """
    Create a Pinecone index if it doesn't already exist.
    Dimension 1536 matches OpenAI text-embedding-3-small output size.
    """
    existing = [i.name for i in pc.list_indexes()]
    if INDEX_NAME not in existing:
        print(f"Creating Pinecone index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc.Index(INDEX_NAME)


# ── Step 4: Ingest Documents ──────────────────────────────────────────────────

def ingest_documents(file_paths: list[str]):
    """
    Full ingestion pipeline:
    Load → Chunk → Embed → Upsert to Pinecone
    """
    index = setup_index()
    documents = load_documents(file_paths)

    if not documents:
        print("No documents loaded.")
        return

    vectors = []
    for doc in documents:
        chunks = chunk_text(doc["text"])
        print(f"Processing '{doc['filename']}' → {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            vector_id = f"{doc['filename']}_chunk_{i}"
            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "text": chunk,
                    "source": doc["filename"],
                    "chunk_index": i
                }
            })

    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)

    print(f"\nSuccessfully ingested {len(vectors)} chunks into Pinecone.")


# ── Step 5: Retrieval + Generation ───────────────────────────────────────────

def answer_question(question: str) -> str:
    """
    Core RAG pipeline:
    1. Embed the question
    2. Search Pinecone for similar chunks
    3. Build a prompt with retrieved context
    4. Get answer from GPT
    """
    index = pc.Index(INDEX_NAME)

    # Embed the question
    question_embedding = get_embedding(question)

    # Retrieve top-k most similar chunks
    results = index.query(
        vector=question_embedding,
        top_k=TOP_K,
        include_metadata=True
    )

    if not results["matches"]:
        return "I couldn't find relevant information to answer your question."

    # Build context from retrieved chunks
    context_parts = []
    for match in results["matches"]:
        source = match["metadata"]["source"]
        text = match["metadata"]["text"]
        score = round(match["score"], 3)
        context_parts.append(f"[Source: {source} | Similarity: {score}]\n{text}")

    context = "\n\n---\n\n".join(context_parts)

    # Prompt engineering: instruct GPT to use only retrieved context
    system_prompt = """You are a helpful QA assistant. 
Answer the user's question using ONLY the provided context.
If the context doesn't contain enough information, say so clearly.
Always cite which source document your answer comes from."""

    user_prompt = f"""Context:
{context}

Question: {question}

Answer:"""

    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2  # Low temperature = more factual, less creative
    )

    return response.choices[0].message.content


# ── CLI Interface ─────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("   RAG QA Bot — by Tharuna Rajpurohit")
    print("=" * 50)
    print("\nCommands:")
    print("  ingest <file1> <file2> ...  — Load documents into Pinecone")
    print("  ask                         — Start asking questions")
    print("  quit                        — Exit\n")

    while True:
        command = input(">> ").strip()

        if command.startswith("ingest "):
            files = command.replace("ingest ", "").split()
            ingest_documents(files)

        elif command == "ask":
            print("\nAsk questions about your documents. Type 'back' to return.\n")
            while True:
                question = input("Question: ").strip()
                if question.lower() == "back":
                    break
                if question:
                    print("\nSearching...\n")
                    answer = answer_question(question)
                    print(f"Answer: {answer}\n")

        elif command in ("quit", "exit", "q"):
            print("Bye!")
            break

        else:
            print("Unknown command. Try: ingest <files> | ask | quit")


if __name__ == "__main__":
    main()
