"""Turn a PDF on disk into page-level text ready for chunking."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from pypdf import PdfReader


@dataclass
class PageText:
    page_number: int
    text: str


def load_pdf_pages(path: str | Path) -> List[PageText]:
    reader = PdfReader(str(path))
    pages: List[PageText] = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(PageText(page_number=i + 1, text=text))
    return pages
