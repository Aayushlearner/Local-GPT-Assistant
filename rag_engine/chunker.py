# rag_engine/chunker.py

from typing import List


def chunk_text(
    text: str,
    chunk_size: int = 350,
    overlap: int = 50,
) -> List[str]:
    """
    Split text into overlapping chunks based on number of words.

    Args:
        text: Full document text.
        chunk_size: Target words per chunk.
        overlap: Number of overlapping words between chunks.

    Returns:
        List of chunk strings.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words).strip()
        if chunk:
            chunks.append(chunk)
        # Move start forward with overlap
        start = end - overlap

        if start < 0:
            start = 0

        if start >= len(words):
            break

    return chunks
