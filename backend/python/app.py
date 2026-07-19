from fastapi import FastAPI
from pydantic import BaseModel

from rag import process_pdf, ask_question

app = FastAPI()


# -----------------------------
# Request Models
# -----------------------------
class PDFRequest(BaseModel):
    path: str


class QueryRequest(BaseModel):
    question: str


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def home():
    return {
        "message": "Python RAG API Running"
    }


@app.post("/process-pdf")
def process_uploaded_pdf(request: PDFRequest):
    try:
        chunks = process_pdf(request.path)

        return {
            "success": True,
            "message": "PDF processed successfully",
            "chunks": chunks
        }

    except Exception as e:
        print("PDF Processing Error:", e)

        return {
            "success": False,
            "error": str(e)
        }


@app.post("/query")
def query_pdf(request: QueryRequest):
    try:
        answer = ask_question(request.question)

        return {
            "success": True,
            "answer": answer
        }

    except Exception as e:
        print("Query Error:", e)

        return {
            "success": False,
            "error": str(e)
        }