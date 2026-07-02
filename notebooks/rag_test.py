import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic

# 1. Load embedding model (downloads ~80MB first run)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Your "documents" — replace with real text later
docs = [
    "The mitochondria is the powerhouse of the cell.",
    "Python's GIL limits true multithreading for CPU-bound work.",
    "Chroma is an open-source embedding database.",
    "Cosine similarity measures the angle between two vectors.",
    "The cell membrane controls what enters and leaves the cell.",
    "The mitochondria generates ATP through cellular respiration.",
    "ATP is the energy currency of the cell."
]

# 3. Chunking: skipped here since docs are short.
#    For real files, split into ~200-500 token chunks with overlap.

# 4. Vector store (in-memory; use PersistentClient to save to disk)
client = chromadb.Client()


collection = client.create_collection("experiment")

collection.add(
    documents=docs,
    embeddings=embedder.encode(docs).tolist(),
    ids=[f"doc_{i}" for i in range(len(docs))],
)

# 5. Query
def ask_chroma_db(question, k=2):
    q_emb = embedder.encode([question]).tolist()
    hits = collection.query(query_embeddings=q_emb, n_results=k)
    context = "\n".join(hits["documents"][0])

    anthropic = Anthropic()  # reads ANTHROPIC_API_KEY from env
    msg = anthropic.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": f"Answer using only this context:\n{context}\n\nQuestion: {question}",
        }],
    )
    return msg.content[0].text

# Encode all docs into a matrix: shape (num_docs, embedding_dim)
doc_matrix = embedder.encode(docs)

# Normalize each row to length 1, so dot product == cosine similarity
def normalize(m):
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / norms

doc_matrix = normalize(doc_matrix)

def ask_numpy(question, k=2):
    # Encode and normalize the query: shape (1, embedding_dim)
    q_vec = normalize(embedder.encode([question]))

    # One matrix multiply gives a similarity score per doc
    # (num_docs, dim) @ (dim, 1) -> (num_docs, 1) -> flatten to (num_docs,)
    scores = doc_matrix @ q_vec.T
    scores = scores.flatten()

    # Indices of the top-k highest scores
    top_idx = np.argsort(scores)[::-1][:k]
    context = "\n".join(docs[i] for i in top_idx)

    anthropic = Anthropic()
    msg = anthropic.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": f"Answer using only this context:\n{context}\n\nQuestion: {question}",
        }],
    )
    return msg.content[0].text

while True:
    question = input("Ask a question (or 'exit' to quit): ")
    if question.lower() == "exit":
        break
    print(ask_numpy(question))
    #print(ask_chroma_db(question))

#print(ask_numpy("What database stores embeddings?"))

#print(ask_chroma_db("What database stores embeddings?"))