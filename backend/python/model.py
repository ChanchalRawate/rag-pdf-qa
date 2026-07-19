import torch

from sentence_transformers import SentenceTransformer

# ======================================================
# Embedding Model
# ======================================================

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

embedder = SentenceTransformer(EMBEDDING_MODEL)



