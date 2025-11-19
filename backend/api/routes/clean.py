"""
Data Cleaning Endpoints
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List
from pathlib import Path

from backend.core.data_loader import DataLoader
from backend.core.data_profiler import DataProfiler
from backend.core.data_cleaner import SmartDataCleaner

router = APIRouter()

UPLOAD_DIR = Path("tmp/uploads")


class CleaningRequest(BaseModel):
    """Request model for executing cleaning"""
    file_id: str
    suggestions_to_apply: List[str]


@router.get("/clean/suggestions/{file_id}")
async def get_cleaning_suggestions(file_id: str):
    """
    Get smart cleaning suggestions for uploaded file

    Args:
        file_id: Identifier of uploaded file

    Returns:
        List of cleaning suggestions with explanations
    """
    try:
        file_path = UPLOAD_DIR / file_id

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        # Load data
        loader = DataLoader(str(file_path))
        df = loader.load()

        # Generate profile (for context)
        profiler = DataProfiler(df)
        profile = profiler.generate_profile()

        # Analyze and get suggestions
        cleaner = SmartDataCleaner(df, profile)
        suggestions = cleaner.analyze_and_suggest()

        return {
            "status": "success",
            "file_id": file_id,
            "suggestions_count": len(suggestions),
            "suggestions": suggestions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clean/execute")
async def execute_cleaning(request: CleaningRequest):
    """
    Execute selected cleaning suggestions

    Args:
        request: CleaningRequest with file_id and suggestions to apply

    Returns:
        Cleaned data profile and saved file info
    """
    try:
        file_path = UPLOAD_DIR / request.file_id

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        # Load data
        loader = DataLoader(str(file_path))
        df = loader.load()

        # Get suggestions
        cleaner = SmartDataCleaner(df)
        cleaner.analyze_and_suggest()

        # Execute cleaning
        cleaned_df = cleaner.execute_cleaning(request.suggestions_to_apply)

        # Save cleaned data
        cleaned_file_path = UPLOAD_DIR / f"cleaned_{request.file_id}"
        cleaned_df.to_csv(cleaned_file_path, index=False)

        # Generate profile of cleaned data
        profiler = DataProfiler(cleaned_df)
        cleaned_profile = profiler.generate_profile()

        return {
            "status": "success",
            "cleaned_file_id": f"cleaned_{request.file_id}",
            "original_rows": len(df),
            "cleaned_rows": len(cleaned_df),
            "applied_suggestions": request.suggestions_to_apply,
            "cleaned_profile": cleaned_profile
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clean/export-script")
async def export_cleaning_script(request: CleaningRequest):
    """
    Generate Python script for the cleaning pipeline

    Args:
        request: CleaningRequest with file_id and suggestions to apply

    Returns:
        Python script as downloadable file
    """
    try:
        file_path = UPLOAD_DIR / request.file_id

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        # Load data
        loader = DataLoader(str(file_path))
        df = loader.load()

        # Get suggestions
        cleaner = SmartDataCleaner(df)
        cleaner.analyze_and_suggest()

        # Generate script
        script = cleaner.generate_cleaning_script(request.suggestions_to_apply)

        # Return as downloadable Python file
        return Response(
            content=script,
            media_type="text/x-python",
            headers={
                "Content-Disposition": f"attachment; filename=clean_{request.file_id.replace('.csv', '')}.py"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
