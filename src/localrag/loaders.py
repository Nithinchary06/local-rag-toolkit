"""Load text, Markdown, and PDF files into Document objects."""

from pathlib import Path
from .models import Document

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf"}


def load_documents(path: str | Path) -> list[Document]:
    """Load one supported file or every supported file below a directory."""
    target = Path(path).expanduser()
    if not target.exists():
        raise FileNotFoundError(f"Path does not exist: {target}")
    files = [target] if target.is_file() else sorted(p for p in target.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
    documents: list[Document] = []
    for file_path in files:
        suffix = file_path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            continue
        if suffix == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise RuntimeError("PDF support requires pypdf. Install it with: pip install pypdf") from exc
            reader = PdfReader(str(file_path))
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    documents.append(Document(text, str(file_path), {"page": page_number}))
        else:
            text = file_path.read_text(encoding="utf-8", errors="replace")
            if text.strip():
                documents.append(Document(text, str(file_path)))
    if target.is_file() and target.suffix.lower() not in SUPPORTED_EXTENSIONS:
        extension = target.suffix or "[no extension]"
        raise ValueError(f"Unsupported file type: {extension}. Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
    if not documents:
        raise ValueError(f"No readable content found at: {target}")
    return documents
