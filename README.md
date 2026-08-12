
````
 RAG-Based PDF Question Answering System

An AI-powered full-stack application that allows users to upload PDF documents and ask natural-language questions about their content.

The application uses a Retrieval-Augmented Generation (RAG) pipeline to retrieve relevant information from the uploaded document and generate context-aware answers using an LLM.

---

  Features

- Upload PDF documents through a React interface
- Extract and preprocess PDF text
- Split documents into smaller chunks
- Generate semantic embeddings using Sentence Transformers
- Store and search embeddings using FAISS
- Retrieve the most relevant document chunks for a question
- Generate context-aware answers using Groq LLM
- RESTful API architecture
- Spring Boot gateway between frontend and AI service
- FastAPI-based AI/RAG service
- Real-time question answering
- CORS-enabled frontend-backend communication

---

  System Architecture

```text
                    ┌─────────────────────┐
                    │    React Frontend   │
                    │                     │
                    │ PDF Upload + Chat   │
                    └──────────┬──────────┘
                               │
                               │ HTTP Request
                               ▼
                    ┌─────────────────────┐
                    │   Spring Boot       │
                    │      Backend        │
                    │                     │
                    │ REST API Gateway    │
                    └──────────┬──────────┘
                               │
                               │ HTTP Request
                               ▼
                    ┌─────────────────────┐
                    │      FastAPI        │
                    │     AI Service      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     RAG Pipeline    │
                    │                     │
                    │ PDF Extraction      │
                    │ Text Cleaning       │
                    │ Text Chunking       │
                    │ Embeddings          │
                    │ FAISS Retrieval     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      Groq LLM       │
                    │                     │
                    │ Answer Generation   │
                    └─────────────────────┘
````

---

##  RAG Pipeline

The application follows these steps:

### 1. PDF Upload

The user uploads a PDF through the React frontend.

```text
React
  ↓
POST /upload-pdf
  ↓
Spring Boot
  ↓
FastAPI
```

### 2. PDF Processing

FastAPI extracts the text from the uploaded PDF and cleans the extracted content.

### 3. Text Chunking

The document is divided into smaller chunks to make semantic retrieval more effective.

### 4. Embedding Generation

Each text chunk is converted into a vector representation using:

```text
BAAI/bge-small-en-v1.5
```

### 5. FAISS Vector Store

The generated embeddings are stored in a FAISS index.

When a user asks a question, the question is also converted into an embedding and compared against the stored vectors.

### 6. Similarity Search

The top relevant chunks are retrieved from FAISS.

### 7. LLM Generation

The retrieved chunks are passed as context to the Groq LLM.

The model generates an answer using only the retrieved PDF context.

---

## 🛠️ Tech Stack

### Frontend

* React.js
* Axios
* HTML/CSS

### Backend

* Java
* Spring Boot
* Spring Web
* RestClient

### AI Service

* Python
* FastAPI
* Sentence Transformers
* FAISS
* Groq API

### Machine Learning / NLP

* BAAI/bge-small-en-v1.5
* Semantic embeddings
* Vector similarity search
* Retrieval-Augmented Generation (RAG)

---

##  Project Structure

```text
pdf-chat-app/
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatBox.jsx
│   │   │   └── FileUpload.jsx
│   │   └── ...
│   └── package.json
│
├── backend/
│   │
│   ├── python/
│   │   ├── app.py
│   │   ├── rag.py
│   │   ├── utils.py
│   │   ├── vector_store.py
│   │   ├── llm.py
│   │   └── requirements.txt
│   │
│   └── spring-backend/
│       ├── src/
│       │   └── main/
│       │       ├── java/
│       │       │   └── com/chanchal/rag_backend/
│       │       │       ├── client/
│       │       │       ├── config/
│       │       │       ├── controller/
│       │       │       ├── dto/
│       │       │       └── service/
│       │       └── resources/
│       │           └── application.properties
│       │
│       ├── pom.xml
│       └── mvnw
│
├── .gitignore
└── README.md
```

---

## 🔌 API Endpoints

### Spring Boot Backend

#### Health Check

```http
GET /
```

Response:

```text
Spring Boot Backend Running!
```

#### Upload PDF

```http
POST /upload-pdf
```

Request:

```text
multipart/form-data
pdf=<PDF file>
```

#### Ask Question

```http
POST /query
```

Request:

```json
{
  "question": "What is regularization?"
}
```

---

## ⚙️ Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/ChanchalRawate/rag-pdf-qa.git
cd rag-pdf-qa
```

---

### 2. Start FastAPI

Navigate to:

```bash
cd backend/python
```

Create and activate the virtual environment:

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

Start FastAPI:

```powershell
uvicorn app:app --reload
```

FastAPI will run on:

```text
http://127.0.0.1:8000
```

---

### 3. Start Spring Boot

Open another terminal:

```powershell
cd backend/spring-backend
```

Run:

```powershell
.\mvnw.cmd spring-boot:run
```

Spring Boot will run on:

```text
http://localhost:8080
```

---

### 4. Start React

Open another terminal:

```powershell
cd frontend
```

Install dependencies:

```powershell
npm install
```

Start the frontend:

```powershell
npm run dev
```

---

##  Environment Variables

The Groq API key is stored locally in:

```text
backend/python/.env
```

Example:

```env
GROQ_API_KEY=your_api_key
```

The `.env` file is intentionally excluded from Git using `.gitignore`.

**Never commit API keys or other secrets to GitHub.**

---

##  Request Flow

### PDF Upload

```text
User
 ↓
React FileUpload
 ↓
Spring Boot /upload-pdf
 ↓
FastAPI /upload-pdf
 ↓
PDF Extraction
 ↓
Text Chunking
 ↓
Embeddings
 ↓
FAISS Index
```

### Question Answering

```text
User Question
 ↓
React ChatBox
 ↓
Spring Boot /query
 ↓
FastAPI /query
 ↓
Question Embedding
 ↓
FAISS Similarity Search
 ↓
Top-K Relevant Chunks
 ↓
Groq LLM
 ↓
Generated Answer
 ↓
React UI
```

---

##  Example

Question:

```text
What is regularization, and how do L1 and L2 differ?
```

The system retrieves the most relevant chunks from the uploaded PDF and passes them to the LLM as context.

The model generates an answer based only on the retrieved document content.

---

## Project Highlights

* Implemented a complete Retrieval-Augmented Generation pipeline
* Integrated React, Spring Boot, and FastAPI into a microservices-style architecture
* Implemented semantic search using Sentence Transformer embeddings and FAISS
* Integrated Groq LLM for context-aware answer generation
* Designed Spring Boot as an API gateway between the frontend and AI service
* Implemented multipart PDF upload and REST API communication
* Added environment-based API key management

---

##  Future Improvements

* Persistent vector database instead of in-memory FAISS storage
* Support for multiple PDFs
* User authentication
* Conversation history
* Streaming LLM responses
* Improved chunking and retrieval strategies
* Cloud deployment
* Docker containerization

---

## 👩‍💻 Author

**Chanchal Rawate**

GitHub:
[https://github.com/ChanchalRawate](https://github.com/ChanchalRawate)






