"""
RAG QA Bot — Retrieval Augmented Generation Question Answering System
Author: Tharuna Rajpurohit
GitHub: github.com/Era-Of-Tharuna

Origin Story:
    My father runs a large wholesale business. Every day they receive hundreds
    of messages from customers asking about product availability, prices, and
    orders. It was impossible to respond to all of them manually and quickly.

    I built this RAG bot so that any business document — price lists, product
    catalogs, FAQs — could be uploaded, and customers or staff could ask
    questions in plain English and get instant, accurate answers.

How it works:
    1. Load documents (PDF or text files — price lists, catalogs, FAQs)
    2. Split them into small overlapping chunks
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
CHUNK_OVERLAP = 50    # overlap between chunks to preserve context at boundaries
TOP_K = 3             # number of similar chunks to retrieve per query


# ── Step 1: Chunking ──────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split a long document into overlapping chunks.

    Why overlap? When a sentence is cut at a chunk boundary, context is lost.
    Overlap ensures neighbouring chunks share some content, so retrieval
    doesn't miss answers that sit at boundaries.
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
    Returns list of {filename, text} dicts.

    Supports:
        .txt  — plain text files (price lists, FAQs, product info)
        .pdf  — PDF documents (catalogs, invoices)
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
    Convert text into a 1536-dimension vector using OpenAI embeddings.

    Why embeddings? Regular keyword search fails when the customer asks
    "do you have 50kg rice bags?" and the document says "50 kilogram rice
    available in bulk". Embeddings capture semantic meaning, so both phrases
    map to nearby points in vector space — and the search finds it anyway.
    """
    response = openai_client.embeddings.create(
        input=text,
        model=EMBED_MODEL
    )
    return response.data[0].embedding


# ── Step 3: Pinecone Setup ────────────────────────────────────────────────────

def setup_index():
    """
    Create Pinecone index if it doesn't already exist.
    Dimension 1536 matches OpenAI text-embedding-3-small output.
    Cosine similarity finds chunks whose meaning is closest to the query.
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
        Load files → Chunk text → Embed each chunk → Upsert to Pinecone

    Run this once per document (or when documents are updated).
    For a wholesale business: run whenever price list or catalog changes.
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

    # Upsert in batches of 100 (Pinecone limit)
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
        2. Search Pinecone for the 3 most similar chunks
        3. Build a prompt with retrieved context
        4. Get a grounded answer from GPT

    Why not just ask GPT directly?
        GPT alone doesn't know your specific business data — product names,
        prices, stock levels. RAG grounds the answer in YOUR documents so
        GPT gives accurate, specific answers instead of hallucinating.
    """
    index = pc.Index(INDEX_NAME)

    # Embed the question using the same model used for documents
    question_embedding = get_embedding(question)

    # Retrieve top-k most similar chunks from Pinecone
    results = index.query(
        vector=question_embedding,
        top_k=TOP_K,
        include_metadata=True
    )

    if not results["matches"]:
        return "I could not find relevant information to answer your question."

    # Build context string from retrieved chunks
    context_parts = []
    for match in results["matches"]:
        source = match["metadata"]["source"]
        text = match["metadata"]["text"]
        score = round(match["score"], 3)
        context_parts.append(f"[Source: {source} | Relevance: {score}]\n{text}")

    context = "\n\n---\n\n".join(context_parts)

    # System prompt: strict instruction to only use provided context
    system_prompt = """You are a helpful business assistant.
Answer the user's question using ONLY the provided context from business documents.
If the context does not contain enough information, say so clearly.
Always mention which document your answer comes from."""

    user_prompt = f"""Context from business documents:
{context}

Customer question: {question}

Answer:"""

    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2  # Low temperature = factual, consistent answers
    )

    return response.choices[0].message.content


# ── CLI Interface ─────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("   RAG QA Bot — by Tharuna Rajpurohit")
    print("   Built to handle business document queries at scale")
    print("=" * 55)
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
                    print("\nSearching documents...\n")
                    answer = answer_question(question)
                    print(f"Answer: {answer}\n")

        elif command in ("quit", "exit", "q"):
            print("Bye!")
            break

        else:
            print("Unknown command. Try: ingest <files> | ask | quit")


if __name__ == "__main__":
    main()
