# 📄 RAG PDF Question Answering System

An AI-powered PDF Question Answering application that allows users to upload a PDF and ask questions in natural language. The system retrieves relevant content using FAISS and generates context-aware answers using a local LLM (Qwen2.5:3B via Ollama).

## 🚀 Features

- Upload PDF documents
- Ask questions about uploaded PDFs
- Retrieval-Augmented Generation (RAG)
- FAISS Vector Search
- Local LLM using Ollama (Qwen2.5:3B)
- React Frontend
- Node.js Backend
- FastAPI for AI processing

## 🛠️ Tech Stack

### Frontend
- React
- Vite

### Backend
- Node.js
- Express.js

### AI Backend
- FastAPI
- FAISS
- Sentence Transformers
- Ollama
- Qwen2.5:3B

## 📂 Project Structure

```
pdf-chat-app/
│
├── frontend/
├── backend/
│   ├── server.js
│   └── python/
└── README.md
```

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/ChanchalRawate/rag-pdf-qa.git
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Backend

```bash
cd backend
npm install
node server.js
```

### FastAPI

```bash
cd backend/python

python -m venv venv

# Windows
venv\Scripts\activate

pip install -r requirements.txt

uvicorn app:app --reload
```

### Ollama

Install Ollama and download the model:

```bash
ollama pull qwen2.5:3b
ollama serve
```

## 📌 Workflow

1. Upload a PDF
2. Extract text
3. Chunk the document
4. Generate embeddings
5. Store embeddings in FAISS
6. Retrieve relevant chunks
7. Generate answers using Qwen2.5:3B

## 👩‍💻 Author

**Chanchal Rawate**
