import streamlit as st
import re
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from PyPDF2 import PdfReader

# ================================
#  Page Config
# ================================
st.set_page_config(
    page_title="RAG PDF QA",
    page_icon="🤖",
    layout="wide"
)

# ================================
#  PDF Loader
# ================================
def load_pdf(file):
    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"

    return text


# ================================
#  Clean Text
# ================================
def clean_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ================================
#  Chunking
# ================================
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
def chunk_text(text, max_tokens=256, overlap=50):
    tokens = tokenizer.encode(text)
    chunks = []

    step = max_tokens - overlap

    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i:i+max_tokens]
        chunk = tokenizer.decode(chunk_tokens)
       
         if len(chunk.strip()) > 20:
            chunks.append(chunk.strip())
             
    return chunks


# ================================
# Retrieval
# ================================
def retrieve(query, index, documents, k=5):
    query_vec = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)

    scores, indices = index.search(query_vec, k)
    return [(documents[i], scores[0][idx]) for idx, i in enumerate(indices[0])]

# ================================
# Re-ranking
# ================================
def rerank(retrieved_chunks):
    return sorted(retrieved_chunks, key=lambda x: x[1], reverse=True)
# ================================
# RAG Pipeline
# ================================
def rag_pipeline(query, index, documents, top_k=5, final_k=3):

    # Step 1: Retrieve
    retrieved = retrieve(query, index, documents, k=top_k)

    # Step 2: Re-rank
    reranked = rerank(retrieved)

    # Step 3: Select context
    selected_chunks = [chunk for chunk, _ in reranked[:final_k]]
    context = " ".join(selected_chunks)
    context = re.sub(r"\s+", " ", context)

    # Step 4: Prompt
    prompt = f"""
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


# ================================
#  UI STARTS HERE
# ================================

st.title("📄 RAG-based PDF Question Answering")

st.markdown("""
Upload a PDF and ask questions.  
The system retrieves relevant content and generates accurate answers.
""")

# ================================
# ⚙️ Sidebar
# ================================
with st.sidebar:
    st.header("⚙️ Settings")

    k = st.slider("Top K Chunks", 1, 10, 5)

    st.markdown("---")
    st.info("Built using RAG + FAISS + Transformers")


# ================================
# 📂 Upload PDF
# ================================
uploaded_file = st.file_uploader("📂 Upload your PDF", type=["pdf"])

if uploaded_file:
    st.success("✅ PDF uploaded successfully!")

    with st.spinner("Processing PDF..."):
        pdf_text = load_pdf(uploaded_file)
        pdf_text = clean_text(pdf_text)
        documents = chunk_text(pdf_text)

        st.write(f"📊 Total Chunks: {len(documents)}")

        # ================================
        # Embeddings + Index (Cosine Similarity)
        # ================================
        embedder = SentenceTransformer("all-MiniLM-L6-v2")

        doc_embeddings = embedder.encode(documents, convert_to_numpy=True)
        faiss.normalize_L2(doc_embeddings)

        dimension = doc_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(doc_embeddings)

        # ================================
        # Model
        # ================================
       
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    # ================================
    # ❓ Question Input
    # ================================
    query = st.text_input("❓ Ask a question from the PDF")

    if query:
        with st.spinner("Generating answer..."):
            answer, chunks = rag_pipeline(query, index, documents, top_k=k)

        # ================================
        # 🤖 Answer Display
        # ================================
        st.markdown("### 🤖 Answer")
        st.success(answer)

        # ================================
        # 🔍 Context Viewer
        # ================================
        with st.expander("🔍 See Retrieved Context"):
            for i, chunk in enumerate(chunks):
                st.write(f"Chunk {i+1}:")
                st.write(chunk)
                st.markdown("---")


# ================================
# Footer
# ================================
st.markdown("---")
st.caption("Built by Chanchal | RAG Project")
