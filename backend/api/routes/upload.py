"""
File Upload Endpoint
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.core.data_loader import DataLoader
import os
from pathlib import Path

router = APIRouter()

# Configure upload directory
UPLOAD_DIR = Path("tmp/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a data file (CSV, Excel, JSON, Parquet)

    Returns:
        - file_id: Unique identifier for the uploaded file
        - filename: Original filename
        - file_size: Size in bytes
        - file_type: Detected file type
    """
    try:
        # Validate file extension
        allowed_extensions = {".csv", ".xlsx", ".xls", ".json", ".parquet"}
        file_ext = os.path.splitext(file.filename)[1].lower()

        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_ext} not supported. Allowed: {allowed_extensions}"
            )

        # Save file
        file_path = UPLOAD_DIR / file.filename
        contents = await file.read()

        with open(file_path, "wb") as f:
            f.write(contents)

        # Load and validate data
        loader = DataLoader(str(file_path))
        df = loader.load()

        return {
            "status": "success",
            "file_id": file.filename,
            "filename": file.filename,
            "file_size": len(contents),
            "file_type": file_ext,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
