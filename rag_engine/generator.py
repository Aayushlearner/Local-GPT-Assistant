# rag_engine/generator.py

from dotenv import load_dotenv
load_dotenv()  # Ensure .env is loaded

import os
from typing import List, Dict, Any

from groq import Groq

FALLBACK_MESSAGE = "I don't know based on the provided documents."


def build_context_from_chunks(chunks: List[Dict[str, Any]], max_chars: int = 6000) -> str:
    parts = []
    total_len = 0

    for chunk in chunks:
        text = chunk["text"]
        addition = (
            f"[Source: {chunk['metadata'].get('file_name')}, chunk {chunk['metadata'].get('chunk_id')}]\n"
            f"{text}\n\n"
        )

        if total_len + len(addition) > max_chars:
            break

        parts.append(addition)
        total_len += len(addition)

    return "".join(parts)


def generate_answer_with_groq(
    question: str,
    relevant_chunks: List[Dict[str, Any]],
    model_name: str = "llama-3.1-8b-instant",
    temperature: float = 0.1,
) -> str:
    if not relevant_chunks:
        return FALLBACK_MESSAGE

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "Error: GROQ_API_KEY not found. Add it to your .env file."

    client = Groq(api_key=api_key)

    context = build_context_from_chunks(relevant_chunks)

    system_prompt = (
        "You are a RAG assistant. Follow these rules:\n"
        "1. Use ONLY the provided context.\n"
        "2. If the answer is not present, say:\n"
        f'"{FALLBACK_MESSAGE}"\n'
        "3. Do not hallucinate or use external knowledge.\n"
    )

    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Give a precise answer."
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Groq API Error: {e}"
