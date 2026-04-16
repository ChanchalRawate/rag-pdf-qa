#  Retrieval-Augmented Document Question Answering System

##  Overview

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline for answering questions from PDF documents.

Instead of relying solely on a language model, the system:

1. Retrieves relevant document chunks using vector search
2. Generates accurate, context-aware answers using a transformer model

---

##  Features

*  Upload and process PDF documents
*  Semantic search using vector embeddings
*  Context-aware answer generation
*  Real-time querying via interactive UI
*  Explainability with retrieved context display

---

##  Architecture

The system follows a **RAG pipeline**:

1. **Document Ingestion**

   * Extract text using PyPDF2
   * Clean and preprocess text

2. **Chunking**

   * Split text into overlapping chunks
   * Improves retrieval relevance

3. **Embedding Generation**

   * Use Sentence-Transformers to convert text into vectors

4. **Vector Search**

   * Store embeddings in FAISS
   * Retrieve Top-K relevant chunks

5. **Answer Generation**

   * Use FLAN-T5
   * Generate answers based on retrieved context

---

##  Tech Stack

* Python
* FAISS
* Transformers (FLAN-T5)
* Sentence-Transformers
* Streamlit

---

##  How It Works

```text
PDF → Text Extraction → Chunking → Embeddings → FAISS Index
→ Retrieve Top-K Chunks → Generate Answer → Display Output
```

---

##  Key Highlights

* Processed **250+ document chunks** for efficient retrieval
* Improved answer relevance through **context-aware retrieval**
* Tuned Top-K and chunk size to reduce irrelevant responses
* Built an **interactive Streamlit UI** for real-time querying

---

##  Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/rag-pdf-qa.git
cd rag-pdf-qa
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

---

##  Usage

1. Upload a PDF file
2. Adjust Top-K retrieval from sidebar
3. Ask questions related to the document
4. View generated answer and supporting context

---

##  Example Output

* Input: *“What is the main objective of the document?”*
* Output: A concise, context-aware answer generated from relevant sections

---

##  Limitations

* Performance depends on chunk size and Top-K selection
* Large PDFs may increase processing time
* No persistent vector storage (in-memory FAISS index)

---

##  Future Improvements

* Add persistent vector database (e.g., FAISS + disk storage)
* Support multiple PDFs and document collections
* Improve answer generation using larger LLMs
* Add citation highlighting in answers

---

##  Learnings

* Practical implementation of Retrieval-Augmented Generation
* Importance of chunking and retrieval tuning
* Tradeoff between retrieval depth (Top-K) and answer quality
* Building end-to-end AI systems with real-time interaction

---

##  Contact

Feel free to connect or reach out for feedback!

---

##  Tags

`RAG` `NLP` `FAISS` `Transformers` `Streamlit` `LLM` `Information Retrieval`

---

