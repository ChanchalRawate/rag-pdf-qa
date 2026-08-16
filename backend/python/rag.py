from utils import load_pdf, clean_text, chunk_text
from vector_store import create_vector_store, search_vector_store
from llm import generate_answer



documents = []
index = None




def process_pdf(pdf_path):
    """
    Read PDF -> Clean -> Chunk -> Create Embeddings -> Store in FAISS
    """

    global documents
    global index

    text = load_pdf(pdf_path)

    if not text.strip():
        raise Exception("No text found in PDF.")

    text = clean_text(text)

    documents = chunk_text(text)

    print(f"\nCreated {len(documents)} chunks")

    index = create_vector_store(documents)

    print("FAISS Index Created.")

    return len(documents)




def retrieve(question, k=3):
    """
    Retrieve top-k relevant chunks
    """

    global index
    global documents

    if index is None:
        return []

    results = search_vector_store(
        question,
        documents,
        index,
        k,
    )

    print("\n========== Retrieved Chunks ==========\n")

    for i, chunk in enumerate(results):
        print("=" * 60)
        print(f"Chunk {i + 1}")
        print(chunk[:300])
        print()

    return results


def ask_question(question):
    """
    Retrieve context then ask LLM
    """

    global index

    if index is None:
        return "Please upload a PDF first."

    chunks = retrieve(question)


    if not chunks:
        return "I couldn't find the answer in the uploaded PDF."

    context = "\n\n".join(chunks)

    answer = generate_answer(
        context=context,
        question=question,
    )

   
    if not answer.strip():
        return "I couldn't find the answer in the uploaded PDF."

    return answer
