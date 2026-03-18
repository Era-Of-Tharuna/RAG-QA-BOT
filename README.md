# RAG QA Bot

A **Retrieval Augmented Generation (RAG)** question answering system built with OpenAI and Pinecone.
Upload any business document and ask questions in plain English — get accurate, source-cited answers powered by GPT.

---

## The Problem That Inspired This

My father runs a large wholesale business. Every day the team receives **hundreds of messages** from customers asking about product availability, pricing, and orders. Responding manually to all of them was slow, inconsistent, and exhausting.

I built this system so that any business document — price lists, product catalogs, FAQs — could be uploaded once, and staff or customers could ask questions and get instant, accurate answers without needing to search through files manually.

---

## How It Works

```
Business Documents → Chunking → Embeddings → Pinecone (vector DB)
                                                      ↓
Customer Question  → Embedding → Similarity Search → Top 3 Chunks
                                                      ↓
                                GPT-3.5 + Context → Final Answer
```

1. **Load** — Upload `.txt` or `.pdf` files (price lists, catalogs, FAQs)
2. **Chunk** — Split into overlapping 500-character segments
3. **Embed** — Convert each chunk to a 1536-dimension vector using OpenAI
4. **Store** — Upsert vectors + metadata into Pinecone index
5. **Query** — Embed the question, find similar chunks, generate answer with GPT

---

## Why RAG Instead of Just GPT?

GPT alone doesn't know your specific business data — your product names, prices, or stock. Without RAG, it would either say "I don't know" or worse, **hallucinate wrong answers**.

RAG grounds every answer in your actual documents. The answer is always traceable back to a real source.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | OpenAI GPT-3.5-turbo |
| Embeddings | OpenAI text-embedding-3-small (1536 dims) |
| Vector Database | Pinecone (Serverless, cosine similarity) |
| Language | Python 3.10+ |

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/Era-Of-Tharuna/RAG-QA-BOT.git
cd RAG-QA-BOT
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add API keys
```bash
cp .env.example .env
```
Fill in your keys in `.env`:
```
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

Get your keys from:
- OpenAI → https://platform.openai.com/api-keys
- Pinecone → https://app.pinecone.io

### 4. Run
```bash
python rag_bot.py
```

---

## Example Usage

```
>> ingest price_list.txt product_catalog.txt
Processing 'price_list.txt' → 6 chunks
Processing 'product_catalog.txt' → 12 chunks
Successfully ingested 18 chunks into Pinecone.

>> ask
Question: Do you have 50kg rice bags and what is the price?

Searching documents...

Answer: Yes, 50kg rice bags are available at Rs. 2,200 per bag for bulk
orders above 10 units. [Source: price_list.txt | Relevance: 0.921]
```

---

## Project Structure

```
RAG-QA-BOT/
├── rag_bot.py              # Main application — load, embed, retrieve, answer
├── requirements.txt        # Python dependencies
├── sample_document.txt     # Example business document to test with
├── .env.example            # API key template (never commit real .env)
└── .gitignore
```

---

## Key Design Decisions

**Why chunk with overlap?**
When text is split at a boundary, context is lost. A 50-character overlap between adjacent chunks ensures no answer falls through the cracks.

**Why cosine similarity?**
Cosine similarity measures the angle between two vectors — it finds chunks whose *meaning* is similar, regardless of exact word match. So "50kg rice bags" matches "50 kilogram rice in bulk" correctly.

**Why low temperature (0.2)?**
Lower temperature makes GPT more deterministic and factual. For business queries, consistency matters more than creativity.

---

## Author

**Tharuna Rajpurohit**
- GitHub: [@Era-Of-Tharuna](https://github.com/Era-Of-Tharuna)
- LinkedIn: [tharuna-rajpurohit](https://linkedin.com/in/tharuna-rajpurohit)
