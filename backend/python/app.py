from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

from rag import process_pdf, ask_question

app = FastAPI()

# -----------------------------------
# Enable CORS
# -----------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your Vercel URL after deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------
# Upload Folder
# -----------------------------------

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -----------------------------------
# Request Models
# -----------------------------------

class QueryRequest(BaseModel):
    question: str

# -----------------------------------
# Routes
# -----------------------------------

@app.get("/")
def home():
    return {
        "message": "Python RAG API Running"
    }

# -----------------------------------
# Upload PDF
# -----------------------------------

@app.post("/upload-pdf")
async def upload_pdf(pdf: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, pdf.filename)

        with open(file_path, "wb") as f:
            f.write(await pdf.read())

        chunks = process_pdf(file_path)

        return {
            "success": True,
            "message": "PDF uploaded successfully.",
            "chunks": chunks,
        }

    except Exception as e:
        print("Upload Error:", e)

        return {
            "success": False,
            "error": str(e),
        }

# -----------------------------------
# Ask Question
# -----------------------------------

@app.post("/query")
def query_pdf(request: QueryRequest):
    try:
        answer = ask_question(request.question)

        return {
            "success": True,
            "answer": answer,
        }

    except Exception as e:
        print("Query Error:", e)

        return {
            "success": False,
            "error": str(e),
        }