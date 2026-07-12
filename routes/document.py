from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from chunking import chunk_pages
from embeddings import embed_texts
from get_db import get_db
from models import Document, DocumentChunk
from pdf_extraction import PDFExtractionError, extract_text_from_pdf
from schemas import ChunkOut, DocumentChunkingOut, DocumentExtractionOut, DocumentOut, PageTextOut

router = APIRouter(prefix="/documents", tags=["Documents"])

MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB, generous MVP ceiling


async def _read_and_validate_pdf_upload(file: UploadFile) -> bytes:
    """Shared upload validation for /extract and /chunk-preview: content-type/
    extension check, empty-file check, size ceiling check."""
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only application/pdf uploads are supported.",
        )

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Uploaded file must have a .pdf extension.",
        )

    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file is empty.",
        )

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size of {MAX_FILE_SIZE_BYTES} bytes.",
        )

    return file_bytes


@router.post("/extract", response_model=DocumentExtractionOut)
async def extract_document_text(file: UploadFile = File(...)):
    file_bytes = await _read_and_validate_pdf_upload(file)

    try:
        extracted = extract_text_from_pdf(file_bytes, file.filename)
    except PDFExtractionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return DocumentExtractionOut(
        filename=extracted.filename,
        page_count=extracted.page_count,
        pages=[
            PageTextOut(
                page_number=p.page_number,
                text=p.text,
                char_count=p.char_count,
            )
            for p in extracted.pages
        ],
    )


@router.post("/chunk-preview", response_model=DocumentChunkingOut)
async def preview_document_chunks(file: UploadFile = File(...)):
    file_bytes = await _read_and_validate_pdf_upload(file)

    try:
        extracted = extract_text_from_pdf(file_bytes, file.filename)
    except PDFExtractionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    chunks = chunk_pages(extracted.pages)

    return DocumentChunkingOut(
        filename=extracted.filename,
        page_count=extracted.page_count,
        total_chunks=len(chunks),
        chunks=[
            ChunkOut(
                chunk_index=c.chunk_index,
                page_number=c.page_number,
                text=c.text,
                token_count=c.token_count,
                char_count=c.char_count,
            )
            for c in chunks
        ],
    )


@router.post("/", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    file_bytes = await _read_and_validate_pdf_upload(file)

    try:
        extracted = extract_text_from_pdf(file_bytes, file.filename)
    except PDFExtractionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    chunks = chunk_pages(extracted.pages)
    vectors = embed_texts([c.text for c in chunks])

    document = Document(filename=extracted.filename, page_count=extracted.page_count)
    document.chunks = [
        DocumentChunk(
            chunk_index=c.chunk_index,
            page_number=c.page_number,
            content=c.text,
            embedding=vector,
        )
        for c, vector in zip(chunks, vectors)
    ]

    db.add(document)
    await db.commit()

    return DocumentOut(
        id=document.id,
        filename=document.filename,
        page_count=document.page_count,
        chunk_count=len(chunks),
        uploaded_at=document.uploaded_at,
    )
