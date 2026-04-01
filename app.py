import streamlit as st
import re
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from PyPDF2 import PdfReader

# ================================
# 🎯 Page Config
# ================================
st.set_page_config(
    page_title="RAG PDF QA",
    page_icon="🤖",
    layout="wide"
)

# ================================
# 📄 PDF Loader
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
# 🧹 Clean Text
# ================================
def clean_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ================================
# ✂️ Chunking
# ================================
def chunk_text(text, chunk_size=120, overlap=30):
    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i+chunk_size])
        if len(chunk.split()) > 30:
            chunks.append(chunk.strip())

    return chunks


# ================================
# 🔍 Retrieval
# ================================
def retrieve(query, k=5):
    query_vector = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vector, k)
    return [documents[i] for i in indices[0]]


# ================================
# 🤖 RAG Pipeline
# ================================
def rag_pipeline(query):
    context_chunks = retrieve(query, k=k)

    context = " ".join(context_chunks[:3])
    context = re.sub(r"\s+", " ", context)

    prompt = f"""
Extract ONLY one clean sentence that answers the question.

Context:
{context}

Question:
{query}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model.generate(
        **inputs,
        max_new_tokens=60,
        do_sample=False
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # clean answer
    sentences = re.split(r'(?<=[.!?])\s+', answer)
    answer = sentences[0]

    return answer.strip()


# ================================
# 🎨 UI STARTS HERE
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

        # Embeddings
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        doc_embeddings = embedder.encode(documents, convert_to_numpy=True)

        dimension = doc_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(doc_embeddings)

        # Model
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    # ================================
    # ❓ Question Input
    # ================================
    query = st.text_input("❓ Ask a question from the PDF")

    if query:
        with st.spinner("Generating answer..."):
            answer = rag_pipeline(query)

        # ================================
        # 🤖 Answer Display
        # ================================
        st.markdown("### 🤖 Answer")
        st.success(answer)

        # ================================
        # 🔍 Context Viewer (IMPRESSIVE 🔥)
        # ================================
        with st.expander("🔍 See Retrieved Context"):
            chunks = retrieve(query, k=3)
            for i, chunk in enumerate(chunks):
                st.write(f"Chunk {i+1}:")
                st.write(chunk)
                st.markdown("---")


# ================================
# Footer
# ================================
st.markdown("---")
st.caption("🚀 Built by Chanchal | RAG Project")
