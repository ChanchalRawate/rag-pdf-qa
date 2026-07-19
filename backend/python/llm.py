import ollama


def generate_answer(context, question):

    system_prompt = """
You are an AI assistant for Question Answering.

Rules:
1. Answer ONLY using information present in the provided context.
2. Do not use your own knowledge or assumptions.
3. Do not add, remove, or modify facts from the context.
4. Keep important names, numbers, dates, and technical terms exactly as written in the context.
5. If the answer is not explicitly available in the context, reply exactly:
I couldn't find the answer in the uploaded PDF.
6. Answer in 2-5 sentences.
7. Do not repeat the question.
8. Do not summarize the entire context. Only answer what is asked.
9. Do not explain your reasoning.
"""

    user_prompt = f"""
Context:
{context}

Question:
{question}

Answer:
"""

    response = ollama.chat(
        model="qwen2.5:3b",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        options={
            "temperature": 0,
        },
    )

    answer = response["message"]["content"].strip()

    if not answer:
        answer = "I couldn't find the answer in the uploaded PDF."

    return answer