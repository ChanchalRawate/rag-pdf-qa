import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "llama-3.3-70b-versatile"


def generate_answer(context, question):

    system_prompt = """
You are an AI assistant for Question Answering.

Rules:
1. Answer ONLY using information present in the provided context.
2. Never use outside knowledge.
3. Do not add, remove, or modify facts.
4. Keep names, numbers, dates, and technical terms exactly as written.
5. If the answer is not explicitly present in the context, reply EXACTLY:
I couldn't find the answer in the uploaded PDF.
6. Answer in 2-5 sentences.
7. Do not repeat the question.
8. Do not summarize the whole context.
9. Do not explain your reasoning.
"""

    user_prompt = f"""
Context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
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
    )

    answer = response.choices[0].message.content.strip()

    if not answer:
        return "I couldn't find the answer in the uploaded PDF."

    return answer
