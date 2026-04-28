import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ================================
# Load Models
# ================================
embedder = SentenceTransformer("all-MiniLM-L6-v2")

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")


# ================================
# Build Index (Cosine Similarity)
# ================================
def build_index(documents):
    embeddings = embedder.encode(documents, convert_to_numpy=True)

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # inner product = cosine after normalization
    index.add(embeddings)

    return index, embeddings


# ================================
# Retrieve (Top-K)
# ================================
def retrieve(query, index, documents, k=5):
    query_vec = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)

    scores, indices = index.search(query_vec, k)

    results = [(documents[i], scores[0][idx]) for idx, i in enumerate(indices[0])]
    return results


# ================================
# Optional: Re-ranking (lightweight)
# ================================
def rerank(query, retrieved_chunks):
    # Simple score-based re-ranking (can replace with cross-encoder later)
    return sorted(retrieved_chunks, key=lambda x: x[1], reverse=True)


# ================================
# RAG Pipeline
# ================================
def rag_pipeline(query, index, documents, top_k=5, final_k=3):
    
    # Step 1: Retrieve
    retrieved = retrieve(query, index, documents, k=top_k)

    # Step 2: Re-rank
    reranked = rerank(query, retrieved)

    # Step 3: Select top chunks
    selected_chunks = [chunk for chunk, _ in reranked[:final_k]]
    context = " ".join(selected_chunks)

    # Step 4: Prompt (Improved)
    prompt = f"""
You are a QA assistant.

Answer ONLY from the given context.
If the answer is not present, say "Not found in document."

Context:
{context}

Question:
{query}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer.strip(), selected_chunks
