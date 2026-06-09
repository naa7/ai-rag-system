"""
Grounded answer generation via Groq.

Pipeline stage 5: take the retrieved chunks and ask the LLM to answer the
question USING ONLY those chunks. Grounding is enforced by the system prompt
(persona + subject come from the active DomainConfig); source attribution is
appended programmatically from chunk metadata so it can never be omitted or
invented by the model.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

from .config import ACTIVE_CONFIG
from .store import RetrievedChunk

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
NO_INFO = "I don't have enough information on that."

SYSTEM_PROMPT = ACTIVE_CONFIG.system_prompt(NO_INFO)


def _format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        blocks.append(f"[Source {i}: {c.label}]\n{c.text}")
    return "\n\n".join(blocks)


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    """Generate a grounded answer from the retrieved chunks. Returns the answer text."""
    if not chunks:
        return NO_INFO

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_key_here":
        return (
            "[GROQ_API_KEY not set — generation skipped. Retrieval still works; see the "
            "retrieved sources below. Add your key to .env to enable answers.]"
        )

    from groq import Groq

    client = Groq(api_key=api_key)
    user_msg = (
        f"CONTEXT:\n{_format_context(chunks)}\n\n"
        f"QUESTION: {question}\n\n"
        "Answer using only the CONTEXT above."
    )
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    return resp.choices[0].message.content.strip()
