import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_chroma import Chroma


# Load environment variables before the app starts.
load_dotenv()


PDF_FILE = "Introduction_to_Data_and_Data_Science.pdf"
OUTPUT_FILE = "cleaned_text.txt"


def clean_text(text: str) -> str:
    """
    Very basic text cleaning:
    - Remove excessive whitespace
    - Remove repeated newlines
    - Fix broken words across line breaks
    """

    # Remove hyphenation at line breaks (e.g. "data-\nscience" → "datascience")
    text = re.sub(r"-\n", "", text)

    # Replace newlines with spaces
    # text = text.replace("\n", " ")

    # Collapse multiple spaces into one
    # text = re.sub(r"\s+", " ", text)# Normalize spaces and tabs, but keep newlines
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def ingest_pdf() -> None:
    pdf_path = Path(PDF_FILE)
    if not pdf_path.exists():
        raise FileNotFoundError(f"{PDF_FILE} not found in project root")

    # 1) Load PDF (one Document per page)
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    print(type(documents))  # list
    print(type(documents[0]))  # Document
    print(documents[0].page_content[:200])
    print(documents[0].metadata)

    print(f"Loaded {len(documents)} pages")

    # 2) Clean each page but keep metadata (best for RAG later)
    cleaned_documents = []
    for doc in documents:
        cleaned_documents.append(
            Document(
                page_content=clean_text(doc.page_content),
                metadata=doc.metadata,
            )
        )

    # 3) Save cleaned text for inspection (with page markers)
    lines = []
    for doc in cleaned_documents:
        page = doc.metadata.get("page", "unknown")
        lines.append(f"\n--- Page {page} ---\n")
        lines.append(doc.page_content)

    Path(OUTPUT_FILE).write_text("\n".join(lines), encoding="utf-8")

    print(f"Cleaned text written to {OUTPUT_FILE}")
    print("Ingestion step complete ✅")


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing from the environment.")
    print("OpenAI API key loaded.")
    ingest_pdf()


if __name__ == "__main__":
    main()
