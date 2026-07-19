import faiss
import numpy as np

from model import embedder

SIMILARITY_THRESHOLD = 0.45


def create_vector_store(documents):

    embeddings = embedder.encode(
        documents,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = embeddings.astype(np.float32)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    print(f"\n✅ FAISS Index Created with {len(documents)} chunks.")

    return index


def search_vector_store(question, documents, index, k=5):

    query_embedding = embedder.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = query_embedding.astype(np.float32)

    scores, indices = index.search(
        query_embedding,
        k
    )

    retrieved_chunks = []

    print("\n========== Retrieved Chunks ==========\n")

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        if score < SIMILARITY_THRESHOLD:
            continue

        print(f"Score : {score:.4f}")
        print("-" * 60)
        print(documents[idx][:300])
        print()

        retrieved_chunks.append(documents[idx])

    return retrieved_chunks