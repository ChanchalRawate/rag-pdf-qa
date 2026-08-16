from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
print(">>> app.py started")
from rag import process_pdf, ask_question
print(">>> rag imported")
app = FastAPI()
print(">>> FastAPI created")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your Vercel URL after deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)



class QueryRequest(BaseModel):
    question: str



@app.get("/")
def home():
    return {
        "message": "Python RAG API Running"
    }



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
