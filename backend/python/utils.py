import re
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ======================================================
# Load PDF
# ======================================================

def load_pdf(pdf_path):

    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# ======================================================
# Clean Text
# ======================================================

def clean_text(text):

    text = text.replace("\n", " ")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ======================================================
# Chunk Text
# ======================================================

def chunk_text(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_text(text)

    print(f"\nCreated {len(chunks)} chunks")

    return chunks