from dataclasses import dataclass

import pymupdf


class PDFExtractionError(Exception):
    """Raised when the given bytes cannot be opened/parsed as a PDF."""


@dataclass
class PageText:
    page_number: int  # 1-indexed
    text: str
    char_count: int


@dataclass
class ExtractedDocument:
    filename: str
    page_count: int
    pages: list[PageText]


def extract_text_from_pdf(file_bytes: bytes, filename: str) -> ExtractedDocument:
    """
    Open PDF bytes with PyMuPDF and extract plain text per page.

    Raises PDFExtractionError if the bytes cannot be opened as a valid PDF
    (corrupt file, wrong format, zero pages, etc). Does NOT raise on pages
    that simply have no extractable text (e.g. scanned/image-only pages) -
    those come back with text="" and char_count=0.
    """
    try:
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise PDFExtractionError(f"Could not open '{filename}' as a PDF: {exc}") from exc

    try:
        if doc.page_count == 0:
            raise PDFExtractionError(f"'{filename}' has no pages")

        pages: list[PageText] = []
        for index in range(doc.page_count):
            page = doc.load_page(index)
            text = page.get_text()
            pages.append(
                PageText(
                    page_number=index + 1,
                    text=text,
                    char_count=len(text),
                )
            )

        return ExtractedDocument(
            filename=filename,
            page_count=doc.page_count,
            pages=pages,
        )
    finally:
        doc.close()
