# 📄 PDF AI Assistant — RAG-Based Question Answering System

An AI-powered PDF Question Answering application that allows users to upload PDF documents and ask questions in natural language.

The application uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant information from uploaded documents and generate context-aware answers.

The project follows a three-layer architecture:

* **React + Vite** — Frontend
* **Spring Boot** — Backend API Gateway & Authentication
* **FastAPI + Python** — PDF processing and RAG pipeline

---

## ✨ Features

* 📄 Upload PDF documents
* 🤖 Ask natural-language questions about uploaded PDFs
* 🔎 Retrieval-Augmented Generation (RAG)
* 🧠 Semantic document retrieval
* 🔐 JWT-based authentication
* 👤 User login and authentication
* 🚪 Secure logout
* 💬 Interactive chat interface
* ⚡ React-based modern UI
* 🔄 Spring Boot API gateway between frontend and AI service
* 🗄️ H2 database for user information
* 🧩 Modular backend architecture

---

## 🏗️ Architecture

```text
                    ┌──────────────────────┐
                    │      React + Vite    │
                    │      Frontend        │
                    │                      │
                    │  Login               │
                    │  PDF Upload          │
                    │  Chat Interface      │
                    └──────────┬───────────┘
                               │
                               │ HTTP
                               │
                               ▼
                    ┌──────────────────────┐
                    │     Spring Boot      │
                    │      Backend         │
                    │                      │
                    │ JWT Authentication   │
                    │ User Management      │
                    │ API Gateway          │
                    │ H2 Database          │
                    └──────────┬───────────┘
                               │
                               │ HTTP
                               │
                               ▼
                    ┌──────────────────────┐
                    │       FastAPI        │
                    │       Python         │
                    │                      │
                    │ PDF Processing       │
                    │ Text Chunking        │
                    │ Embeddings           │
                    │ Vector Retrieval     │
                    │ RAG Question Answer  │
                    └──────────────────────┘
```

### Request Flow

For authentication:

```text
React
  ↓
POST /auth/login
  ↓
Spring Boot
  ↓
Validate user
  ↓
Generate JWT
  ↓
React stores JWT
```

For PDF upload:

```text
React
  ↓
POST /upload-pdf
  ↓
Spring Boot
  ↓
FastAPI
  ↓
PDF processing
  ↓
Text chunks / embeddings
```

For question answering:

```text
React
  ↓
POST /query
  ↓
Spring Boot
  ↓
FastAPI
  ↓
Retrieve relevant PDF context
  ↓
Generate answer
  ↓
Spring Boot
  ↓
React
```

---

# 🛠️ Tech Stack

## Frontend

* React.js
* Vite
* Axios
* JavaScript
* CSS

## Backend

* Java
* Spring Boot
* Spring Security
* Spring Data JPA
* Maven
* JWT
* H2 Database

## AI / RAG Backend

* Python
* FastAPI
* Uvicorn
* PDF processing libraries
* Sentence Transformers / embeddings
* Vector search
* Retrieval-Augmented Generation

---

# 📁 Project Structure

```text
pdf-chat-app/
│
├── backend/
│   │
│   ├── python/
│   │   ├── app.py
│   │   ├── venv/
│   │   └── ...
│   │
│   └── spring-backend/
│       ├── pom.xml
│       │
│       ├── src/
│       │   ├── main/
│       │   │   ├── java/
│       │   │   │   └── com/
│       │   │   │       └── chanchal/
│       │   │   │           └── rag_backend/
│       │   │   │               │
│       │   │   │               ├── config/
│       │   │   │               │   └── SecurityConfig.java
│       │   │   │               │
│       │   │   │               ├── controller/
│       │   │   │               │   ├── AuthController.java
│       │   │   │               │   └── HomeController.java
│       │   │   │               │
│       │   │   │               ├── dto/
│       │   │   │               │   ├── LoginRequest.java
│       │   │   │               │   ├── RegisterRequest.java
│       │   │   │               │   ├── QueryRequest.java
│       │   │   │               │   └── QueryResponse.java
│       │   │   │               │
│       │   │   │               ├── entity/
│       │   │   │               │   └── User.java
│       │   │   │               │
│       │   │   │               ├── repository/
│       │   │   │               │   └── UserRepository.java
│       │   │   │               │
│       │   │   │               ├── security/
│       │   │   │               │   ├── JwtAuthenticationFilter.java
│       │   │   │               │   └── JwtService.java
│       │   │   │               │
│       │   │   │               └── service/
│       │   │   │                   ├── AuthService.java
│       │   │   │                   ├── CustomUserDetailsService.java
│       │   │   │                   └── QueryService.java
│       │   │   │
│       │   │   └── resources/
│       │   │       └── application.properties
│       │   │
│       │   └── test/
│       │
│       └── ...
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Answer.jsx
│   │   │   ├── ChatBox.jsx
│   │   │   ├── FileUpload.jsx
│   │   │   ├── Login.jsx
│   │   │   ├── Login.css
│   │   │   └── Navbar.jsx
│   │   │
│   │   ├── App.jsx
│   │   ├── App.css
│   │   └── index.css
│   │
│   ├── package.json
│   └── ...
│
├── .gitignore
└── README.md
```

---

# 🔐 Authentication

The application uses **JWT-based authentication** through Spring Security.

### Authentication Flow

1. User enters username and password.
2. React sends credentials to Spring Boot.
3. Spring Boot verifies the user against the H2 database.
4. A JWT is generated after successful authentication.
5. React stores the token in `localStorage`.
6. Protected API requests include the token using:

```text
Authorization: Bearer <JWT_TOKEN>
```

7. `JwtAuthenticationFilter` validates the token.
8. Spring Security allows the request to continue if authentication succeeds.

---

# 🔑 Environment Variables

The JWT secret should **never be committed to GitHub**.

The Spring Boot configuration uses:

```properties
jwt.secret=${JWT_SECRET}
```

Set the secret as an environment variable before starting Spring Boot.

### Windows PowerShell

```powershell
$env:JWT_SECRET="your-new-secret-key"
```

Verify:

```powershell
echo $env:JWT_SECRET
```

Then start Spring Boot.

> Do not put the actual JWT secret inside `application.properties` or commit it to GitHub.

---

# ⚙️ Prerequisites

Make sure you have installed:

* Java 17+
* Maven or Maven Wrapper
* Python 3.x
* Node.js
* npm
* Git

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/ChanchalRawate/rag-pdf-qa.git
```

Move into the project:

```bash
cd rag-pdf-qa
```

---

# 🐍 Running the FastAPI Backend

Open **Terminal 1**.

Navigate to the Python backend:

```powershell
cd backend\python
```

Activate the virtual environment:

```powershell
.\venv\Scripts\activate
```

Start FastAPI:

```powershell
uvicorn app:app --reload --port 8000
```

You should see:

```text
Uvicorn running on http://127.0.0.1:8000
```

FastAPI documentation:

```text
http://localhost:8000/docs
```

---

# ☕ Running the Spring Boot Backend

Open **Terminal 2**.

Navigate to the Spring Boot backend:

```powershell
cd backend\spring-backend
```

Set the JWT secret:

```powershell
$env:JWT_SECRET="your-new-secret-key"
```

Start Spring Boot:

```powershell
.\mvnw.cmd spring-boot:run
```

The backend will run on:

```text
http://localhost:8080
```

---

# ⚛️ Running the React Frontend

Open **Terminal 3**.

Navigate to the frontend:

```powershell
cd frontend
```

Install dependencies:

```powershell
npm install
```

Start the development server:

```powershell
npm run dev
```

The frontend will be available at:

```text
http://localhost:5173
```

---

# 🧪 Running the Complete Application

For the complete application, run all three services.

### Terminal 1 — FastAPI

```powershell
cd backend\python
.\venv\Scripts\activate
uvicorn app:app --reload --port 8000
```

### Terminal 2 — Spring Boot

```powershell
cd backend\spring-backend
$env:JWT_SECRET="your-new-secret-key"
.\mvnw.cmd spring-boot:run
```

### Terminal 3 — React

```powershell
cd frontend
npm run dev
```

Then open:

```text
http://localhost:5173
```

---

# 📄 Using the Application

## 1. Login

Open the application and log in using your registered credentials.

After successful authentication, the JWT token is stored in the browser.

## 2. Upload a PDF

Select a PDF using the **Upload Document** section.

The PDF is sent through:

```text
React → Spring Boot → FastAPI
```

The FastAPI service processes the document and creates the required chunks/embeddings for retrieval.

## 3. Ask Questions

Enter a question such as:

```text
What is regularization?
```

The question follows:

```text
React
 ↓
Spring Boot
 ↓
FastAPI
 ↓
Retriever
 ↓
Relevant PDF context
 ↓
RAG generation
 ↓
Answer
```

The answer is then displayed in the chat interface.

---

# 🔌 API Endpoints

## Spring Boot

### Health Check

```http
GET /
```

Response:

```text
Spring Boot Backend Running!
```

### Login

```http
POST /auth/login
```

Request:

```json
{
  "username": "your_username",
  "password": "your_password"
}
```

Response:

```text
JWT token
```

### PDF Upload

```http
POST /upload-pdf
```

Form-data:

```text
pdf: <PDF file>
```

### Ask a Question

```http
POST /query
```

Request:

```json
{
  "question": "What is regularization?"
}
```

Example response:

```json
{
  "success": true,
  "answer": "Regularization adds a penalty term to the loss function to discourage overly complex models and reduce overfitting."
}
```

---

# 🧠 How RAG Works in This Project

The system follows the basic Retrieval-Augmented Generation pipeline:

```text
PDF
 ↓
Text Extraction
 ↓
Text Chunking
 ↓
Embeddings
 ↓
Vector Storage
 ↓
User Question
 ↓
Question Embedding
 ↓
Similarity Search
 ↓
Relevant Context
 ↓
Answer Generation
```

Instead of asking the language model to answer using only its pre-trained knowledge, the application retrieves relevant information from the uploaded PDF and uses that context to generate the response.

This helps the system answer questions specifically based on the uploaded document.

---

# 🗄️ Database

The Spring Boot backend uses **H2 Database** for user authentication data.

Database configuration:

```properties
spring.datasource.url=jdbc:h2:file:./data/ragdb
spring.datasource.driver-class-name=org.h2.Driver
spring.datasource.username=sa
```

The H2 console is enabled at:

```text
http://localhost:8080/h2-console
```

The local database files are ignored by Git using `.gitignore`.

---

# 🛡️ Security

The project follows basic security practices:

* JWT authentication
* Password-based user authentication
* Protected API endpoints
* JWT secret stored using an environment variable
* Local database files excluded from Git
* Python virtual environment excluded from Git
* Node modules excluded from Git
* Build artifacts excluded from Git

Never commit:

```text
JWT secrets
.env files
Passwords
Database files
Uploaded PDFs
Virtual environments
node_modules
target/
```

---

# 🧹 Git Ignore

The project ignores generated and sensitive files such as:

```text
node_modules/
dist/
venv/
__pycache__/
.env
backend/spring-backend/target/
backend/spring-backend/data/
```

---

# 🔮 Future Improvements

Possible future improvements include:

* 📚 Support for multiple PDFs
* 👤 User-specific document collections
* 💾 Persistent vector database
* 📝 Conversation history
* 📌 Source citations for generated answers
* 📊 Retrieval confidence scores
* 🗑️ Delete uploaded documents
* 🔄 Document re-indexing
* 🌐 Production deployment
* 🐳 Docker containerization
* ☁️ Cloud deployment
* 🔑 Refresh-token authentication
* 🛡️ Improved password hashing and security policies

---

# 📌 Project Status

### Current functionality

*  React frontend
*  PDF upload UI
*  Chat interface
*  FastAPI RAG backend
*  Spring Boot backend gateway
*  JWT authentication
*  Login
*  Logout
*  H2 user database
*  Protected API endpoints
*  PDF question answering
*  Local end-to-end pipeline

---

# 👩‍💻 Author

**Chanchal Rawate**

IIT (ISM) Dhanbad

GitHub:
https://github.com/ChanchalRawate

---

# ⭐ Acknowledgements

This project was developed as a practical implementation of:

* Retrieval-Augmented Generation
* Semantic Search
* Natural Language Processing
* REST APIs
* Spring Security
* JWT Authentication
* React-based web applications

---

## ⭐ If you find this project useful

Give the repository a ⭐ on GitHub and feel free to explore, improve, and extend the project.
