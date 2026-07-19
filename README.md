# 📄 RAG PDF Question Answering System

An AI-powered Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions in natural language. The system retrieves relevant content using FAISS vector search and generates context-aware answers using the Groq API.

---

## 🚀 Features

- 📄 Upload PDF documents
- 💬 Ask questions about uploaded PDFs
- 🔍 Retrieval-Augmented Generation (RAG)
- 📚 Semantic Search using FAISS
- 🤖 Answer generation using Groq API
- ⚡ FastAPI Backend
- ⚛️ React Frontend
- 🧠 Sentence Transformer Embeddings

---

## 🛠️ Tech Stack

### Frontend

- React
- Vite
- Axios

### Backend

- FastAPI
- Python

### AI & Machine Learning

- Groq API
- Sentence Transformers
- FAISS
- PyPDF2
- Hugging Face Embedding Model

---

## 📂 Project Structure

```
pdf-chat-app/
│
├── frontend/
│
├── backend/
│   └── python/
│       ├── app.py
│       ├── rag.py
│       ├── llm.py
│       ├── vector_store.py
│       ├── model.py
│       ├── utils.py
│       └── requirements.txt
│
└── README.md
```

---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/ChanchalRawate/rag-pdf-qa.git

cd rag-pdf-qa
```

---

### 2. Frontend

```bash
cd frontend

npm install

npm run dev
```

---

### 3. Backend

```bash
cd backend/python

python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file inside `backend/python`.

```env
GROQ_API_KEY=YOUR_GROQ_API_KEY
```

### Run FastAPI

```bash
uvicorn app:app --reload
```

---

## 📌 Workflow

1. Upload a PDF
2. Extract text from the PDF
3. Clean and preprocess text
4. Split text into chunks
5. Generate embeddings using Sentence Transformers
6. Store embeddings in FAISS
7. Retrieve the most relevant chunks
8. Send retrieved context to Groq API
9. Generate the final answer

---

# 🚀 Project Evolution

### Version 1 — Hugging Face Local LLM

Initially the project used the local Hugging Face model:

```
Qwen/Qwen2.5-0.5B-Instruct
```

**Challenges**

- Large model download
- Slow installation
- High memory usage
- Difficult deployment

**Decision**

Migrated to Ollama for faster local inference.

---

### Version 2 — Ollama

Local model used:

```
qwen2.5:3b
```

**Advantages**

- Fast local inference
- Better response quality
- No API cost

**Challenges**

- Required local Ollama runtime
- Public deployment required cloud GPU
- Cloud GPU providers required payment information

**Decision**

Migrated to Google Gemini API.

---

### Version 3 — Google Gemini API

The local LLM was replaced with the Gemini API.

**Advantages**

- Lightweight backend
- No local model download
- Easy API integration

**Challenges**

- Model deprecation
- API quota limitations

**Decision**

Migrated to Groq API.

---

### Version 4 — Groq API (Current)

The application now uses the Groq API for answer generation.

**Advantages**

- Very fast inference
- Lightweight architecture
- Easy deployment
- Production-friendly API

---

## 🏗️ Architecture Evolution

### Previous Architecture

```
React
   │
Node.js (Express)
   │
FastAPI
   │
FAISS
   │
Ollama
```

### Current Architecture

```
React
   │
FastAPI
   │
PDF Processing
   │
Sentence Transformers
   │
FAISS
   │
Groq API
```

---

## ⚠️ Deployment Note

The project runs successfully in the local environment.

Deployment on the Render Free Tier was attempted, but the application exceeded the available 512 MB memory because the embedding model relies on PyTorch and Sentence Transformers during startup.

Future improvements include using a lightweight embedding API or deploying on a higher-memory instance.

---

## ⚠️ Challenges Faced

During the development of this project, several technical challenges were encountered and resolved:

### 1. Hugging Face Local Model

- Large model downloads increased setup time.
- High memory consumption made deployment difficult.

### 2. Ollama Integration

- Achieved fast local inference.
- However, Ollama requires a local runtime and is not suitable for free cloud deployment.

### 3. Gemini API Migration

- Replaced the local LLM with the Gemini API.
- Faced model compatibility and API quota limitations.

### 4. Groq API Migration

- Migrated to the Groq API for faster inference and simpler integration.
- Reduced backend complexity by removing local LLM dependencies.

### 5. Backend Simplification

- Initially used both Express.js and FastAPI.
- Removed the Express backend and migrated to a single FastAPI backend to simplify the architecture and reduce unnecessary API calls.

### 6. Deployment Challenges

- Successfully ran the application in the local environment.
- Deployment on Render Free Tier failed because the embedding model (Sentence Transformers + PyTorch) exceeded the available 512 MB memory limit.
- Identified that a lightweight embedding service or a higher-memory deployment environment would be required for cloud deployment.
  
---

## 🚀 Future Improvements

The following enhancements are planned for future versions of the project:

- Deploy the complete application to the cloud using a lightweight embedding service or a higher-memory hosting platform.
- Replace the local embedding model with an embedding API to reduce memory usage and improve deployment compatibility.
- Support multiple PDF uploads and create a searchable document library.
- Store embeddings in a persistent vector database (such as ChromaDB, Pinecone, or Weaviate) instead of rebuilding the FAISS index after every upload.
- Add conversation history to enable context-aware multi-turn question answering.
- Improve retrieval quality using hybrid search and reranking techniques.
- Support additional document formats such as DOCX and TXT.
- Implement user authentication and personal document management.
- Add source citations by highlighting the exact PDF sections used to generate each answer.
- Containerize the application with Docker and automate deployment using CI/CD pipelines.

## 👩‍💻 Author

**Chanchal Rawate**
