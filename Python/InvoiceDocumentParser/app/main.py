from __future__ import annotations

import asyncio
import logging
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from app.services.image_service import ImageService
from app.services.ocr_service import OcrService
from app.services.tnote_contract import to_pascal_keys


BASE_DIRECTORY = Path(__file__).resolve().parent.parent
UPLOAD_DIRECTORY = BASE_DIRECTORY / "uploads"
logger = logging.getLogger("uvicorn.error")

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/tiff",
    "image/bmp",
    "image/webp",
    "image/gif",
    "application/pdf",
}

ALLOWED_EXTENSIONS = ImageService.SUPPORTED_EXTENSIONS
GENERIC_BINARY_CONTENT_TYPES = {None, "application/octet-stream"}

MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024
OCR_BASIC_USERNAME_ENV = "OCR_BASIC_USERNAME"
OCR_BASIC_PASSWORD_ENV = "OCR_BASIC_PASSWORD"

ocr_service: OcrService | None = None
basic_auth = HTTPBasic()


def _basic_auth_credentials() -> tuple[str, str]:
    username = os.getenv(OCR_BASIC_USERNAME_ENV, "")
    password = os.getenv(OCR_BASIC_PASSWORD_ENV, "")

    if not username or not password:
        raise RuntimeError(
            "Basic authentication is not configured. Set "
            f"{OCR_BASIC_USERNAME_ENV} and {OCR_BASIC_PASSWORD_ENV}."
        )

    return username, password


def _require_basic_auth(
    credentials: Annotated[HTTPBasicCredentials, Depends(basic_auth)],
) -> None:
    expected_username, expected_password = _basic_auth_credentials()
    username_matches = secrets.compare_digest(
        credentials.username.encode("utf-8"),
        expected_username.encode("utf-8"),
    )
    password_matches = secrets.compare_digest(
        credentials.password.encode("utf-8"),
        expected_password.encode("utf-8"),
    )

    if not (username_matches and password_matches):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials.",
            headers={"WWW-Authenticate": "Basic"},
        )


def _run_ocr_pipeline(source_path: Path) -> dict[str, Any]:
    if ocr_service is None:
        raise RuntimeError("The PaddleOCR service is not initialized.")

    started = time.perf_counter()
    is_pdf = ImageService.is_pdf(source_path)
    source = "pdf" if is_pdf else "image"
    ocr_image, _ = ImageService.prepare_document_array(
        source_path,
        canonical_width=(
            ImageService.PDF_OCR_WIDTH
            if is_pdf
            else ImageService.CANONICAL_OCR_WIDTH
        ),
        maximum_height=(
            ImageService.PDF_OCR_MAX_HEIGHT
            if is_pdf
            else ImageService.CANONICAL_OCR_MAX_HEIGHT
        ),
        pdf_scale=4.0 if is_pdf else 2.0,
    )
    native_items = (
        ImageService.extract_pdf_text_items(
            source_path,
            target_width=ocr_image.shape[1],
            target_height=ocr_image.shape[0],
        )
        if is_pdf
        else []
    )
    result = ocr_service.extract_text_from_image(
        ocr_image,
        native_items=native_items,
        source=source,
    )
    result["processing_time_seconds"] = round(time.perf_counter() - started, 3)
    result["page_count"] = ImageService.page_count(source_path)
    return result


@asynccontextmanager
async def lifespan(_: FastAPI):
    global ocr_service

    # Fail startup rather than accidentally exposing OCR without credentials.
    _basic_auth_credentials()
    UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # Load the OCR model once when the application starts.
    ocr_service = OcrService()

    yield


app = FastAPI(
    title="Invoice Document Parser API",
    description=(
        "Self-hosted invoice and receipt OCR API built using PaddleOCR."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:4200",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "application": "Invoice Document Parser API",
        "status": "Running",
        "documentation": "/docs",
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "Healthy",
        "ocrLoaded": ocr_service is not None,
    }


@app.post(
    "/api/ocr/extract",
    dependencies=[Depends(_require_basic_auth)],
)
async def extract_ocr(
    file: Annotated[
        UploadFile,
        File(description="Invoice or receipt image"),
    ],
) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file does not have a filename.",
        )

    extension = Path(file.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file extension. "
                "Allowed extensions are JPG, JPEG, PNG, TIFF, BMP, WEBP, GIF and PDF."
            ),
        )

    if (
        file.content_type not in ALLOWED_CONTENT_TYPES
        and file.content_type not in GENERIC_BINARY_CONTENT_TYPES
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported content type: {file.content_type}. "
                "Allowed content types are image/jpeg, image/png, image/tiff, "
                "image/bmp, image/webp, image/gif and application/pdf."
            ),
        )

    file_contents = await file.read()

    if not file_contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(file_contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="The file exceeds the maximum size of 15 MB.",
        )

    generated_file_name = f"{uuid.uuid4()}{extension}"
    uploaded_file_path = UPLOAD_DIRECTORY / generated_file_name

    try:
        with uploaded_file_path.open("wb") as destination:
            destination.write(file_contents)

        if ocr_service is None:
            raise RuntimeError("The PaddleOCR service is not initialized.")

        if not ImageService.is_supported(uploaded_file_path):
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type.",
            )

        result = await asyncio.to_thread(
            _run_ocr_pipeline,
            uploaded_file_path,
        )
        logger.info(
            "OCR completed: %d items extracted.",
            len(result.get("items", [])),
        )
        return to_pascal_keys(result)

    except HTTPException:
        raise

    except Exception as exception:
        raise HTTPException(
            status_code=500,
            detail=f"OCR processing failed: {str(exception)}",
        ) from exception

    finally:
        await file.close()

        if uploaded_file_path.exists():
            uploaded_file_path.unlink()
