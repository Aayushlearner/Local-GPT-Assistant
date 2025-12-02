# rag_engine/extractor.py

from typing import BinaryIO
from pathlib import Path

import PyPDF2
import pandas as pd
import docx


def extract_text(file_obj: BinaryIO, filename: str) -> str:
    """
    Extract text from an uploaded file based on its extension.

    Supported:
    - .pdf
    - .txt
    - .csv
    - .docx
    """
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        return _extract_from_pdf(file_obj)
    elif suffix == ".txt":
        return _extract_from_txt(file_obj)
    elif suffix == ".csv":
        return _extract_from_csv(file_obj)
    elif suffix == ".docx":
        return _extract_from_docx(file_obj)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def _extract_from_pdf(file_obj: BinaryIO) -> str:
    reader = PyPDF2.PdfReader(file_obj)
    texts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        texts.append(page_text)
    return "\n".join(texts)


def _extract_from_txt(file_obj: BinaryIO) -> str:
    content = file_obj.read()
    if isinstance(content, bytes):
        return content.decode("utf-8", errors="ignore")
    return str(content)


def _extract_from_csv(file_obj: BinaryIO) -> str:
    df = pd.read_csv(file_obj)
    # Convert entire dataframe to a readable string
    return df.to_string(index=False)


def _extract_from_docx(file_obj: BinaryIO) -> str:
    document = docx.Document(file_obj)
    paragraphs = [para.text for para in document.paragraphs]
    return "\n".join(paragraphs)
