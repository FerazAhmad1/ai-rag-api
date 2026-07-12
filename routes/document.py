from fastapi import APIRouter, File, HTTPException, UploadFile, status

from pdf_extraction import PDFExtractionError, extract_text_from_pdf
from schemas import DocumentExtractionOut, PageTextOut

router = APIRouter(prefix="/documents", tags=["Documents"])

MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB, generous MVP ceiling


@router.post("/extract", response_model=DocumentExtractionOut)
async def extract_document_text(file: UploadFile = File(...)):
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
