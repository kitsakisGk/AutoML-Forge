"""
Data Profiling Endpoint
"""
from fastapi import APIRouter, HTTPException
from backend.core.data_loader import DataLoader
from backend.core.data_profiler import DataProfiler
from pathlib import Path

router = APIRouter()

UPLOAD_DIR = Path("tmp/uploads")


@router.get("/profile/{file_id}")
async def profile_data(file_id: str):
    """
    Generate data profile for uploaded file

    Args:
        file_id: Identifier of uploaded file

    Returns:
        Comprehensive data profile including:
        - Basic statistics
        - Missing values analysis
        - Data types
        - Correlations
        - Unique values
    """
    try:
        file_path = UPLOAD_DIR / file_id

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        # Load data
        loader = DataLoader(str(file_path))
        df = loader.load()

        # Generate profile
        profiler = DataProfiler(df)
        profile = profiler.generate_profile()

        return {
            "status": "success",
            "file_id": file_id,
            "profile": profile
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
