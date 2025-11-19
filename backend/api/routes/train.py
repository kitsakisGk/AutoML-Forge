"""
Model Training Endpoints
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path

from backend.core.data_loader import DataLoader
from backend.core.automl_engine import AutoMLEngine

router = APIRouter()

UPLOAD_DIR = Path("tmp/uploads")


class TrainRequest(BaseModel):
    """Request model for training"""
    file_id: str
    target_column: str
    test_size: float = 0.2


class SHAPRequest(BaseModel):
    """Request model for SHAP explanations"""
    file_id: str
    target_column: str
    model_name: str = None  # If None, uses best model
    max_samples: int = 100


@router.post("/train/start")
async def start_training(request: TrainRequest):
    """
    Start AutoML training

    Args:
        request: TrainRequest with file_id and target_column

    Returns:
        Training results with model comparisons
    """
    try:
        file_path = UPLOAD_DIR / request.file_id

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        # Load data
        loader = DataLoader(str(file_path))
        df = loader.load()

        # Validate target column
        if request.target_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Target column '{request.target_column}' not found in data"
            )

        # Initialize AutoML engine
        engine = AutoMLEngine(
            df=df,
            target_column=request.target_column,
            test_size=request.test_size
        )

        # Detect problem type
        problem_type = engine.detect_problem_type()

        # Prepare data
        prep_info = engine.prepare_data()

        # Train models
        engine.train_models()

        # Get summary
        summary = engine.get_summary()

        return {
            "status": "success",
            "problem_type": problem_type,
            "preparation_info": prep_info,
            "summary": summary
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/train/columns/{file_id}")
async def get_columns(file_id: str):
    """
    Get column names and info from uploaded file

    Args:
        file_id: Identifier of uploaded file

    Returns:
        Column information for target selection
    """
    try:
        file_path = UPLOAD_DIR / file_id

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        # Load data
        loader = DataLoader(str(file_path))
        df = loader.load()

        # Get column info
        columns_info = []
        for col in df.columns:
            col_info = {
                "name": col,
                "dtype": str(df[col].dtype),
                "unique_values": int(df[col].nunique()),
                "missing_count": int(df[col].isnull().sum()),
                "sample_values": df[col].dropna().head(3).tolist()
            }
            columns_info.append(col_info)

        return {
            "status": "success",
            "columns": columns_info,
            "total_columns": len(df.columns),
            "total_rows": len(df)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train/explain")
async def explain_model(request: SHAPRequest):
    """
    Get SHAP explanations for trained model

    Args:
        request: SHAPRequest with file_id, target_column, and optional model_name

    Returns:
        SHAP feature importance and explanations
    """
    try:
        file_path = UPLOAD_DIR / request.file_id

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        # Load data
        loader = DataLoader(str(file_path))
        df = loader.load()

        # Validate target column
        if request.target_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Target column '{request.target_column}' not found in data"
            )

        # Initialize AutoML engine and train
        engine = AutoMLEngine(
            df=df,
            target_column=request.target_column,
            use_mlflow=False  # Don't log during SHAP computation
        )

        engine.detect_problem_type()
        engine.prepare_data()
        engine.train_models()

        # Compute SHAP values
        shap_results = engine.compute_shap_values(
            model_name=request.model_name,
            max_samples=request.max_samples
        )

        if shap_results and "error" in shap_results:
            raise HTTPException(status_code=500, detail=shap_results["error"])

        return {
            "status": "success",
            "shap_explanations": shap_results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
