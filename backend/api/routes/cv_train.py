"""
Computer Vision Training Endpoints
"""
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from pathlib import Path
from typing import Optional
import shutil

from backend.core.image_loader import ImageDataLoader
from backend.core.cv_automl_engine import CVAutoMLEngine
from backend.core.cv_predictor import CVPredictor
import numpy as np
from PIL import Image
import io

router = APIRouter()

UPLOAD_DIR = Path("tmp/cv_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class CVTrainRequest(BaseModel):
    """Request model for CV training"""
    dataset_id: str
    num_epochs: int = 5
    batch_size: int = 32
    test_size: float = 0.2


@router.post("/cv/upload")
async def upload_image_dataset(file: UploadFile = File(...)):
    """
    Upload image dataset (ZIP file with folder-per-class structure)

    Expected structure:
    dataset.zip
        /class1
            image1.jpg
            image2.jpg
        /class2
            image1.jpg
            image2.jpg

    Returns:
        dataset_id for use in training
    """
    try:
        # Generate unique ID
        import uuid
        dataset_id = str(uuid.uuid4())

        # Save uploaded file
        file_path = UPLOAD_DIR / f"{dataset_id}.zip"

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract ZIP
        import zipfile
        extract_dir = UPLOAD_DIR / dataset_id

        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        # Remove ZIP file
        file_path.unlink()

        # Count images
        image_loader = ImageDataLoader(str(extract_dir))
        dataset = image_loader.load()

        return {
            "status": "success",
            "dataset_id": dataset_id,
            "dataset_info": dataset["dataset_info"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cv/train/start")
async def start_cv_training(request: CVTrainRequest):
    """
    Start Computer Vision AutoML training

    Args:
        request: CVTrainRequest with dataset_id and training parameters

    Returns:
        Training results with model comparisons
    """
    try:
        dataset_path = UPLOAD_DIR / request.dataset_id

        if not dataset_path.exists():
            raise HTTPException(status_code=404, detail="Dataset not found")

        # Load dataset
        image_loader = ImageDataLoader(str(dataset_path))
        dataset = image_loader.load()

        # Initialize CV AutoML engine
        engine = CVAutoMLEngine(
            images=dataset["images"],
            labels=dataset["labels"],
            class_names=dataset["class_names"],
            test_size=request.test_size,
            num_epochs=request.num_epochs,
            batch_size=request.batch_size
        )

        # Prepare data
        prep_info = engine.prepare_data()

        # Train models
        engine.train_models()

        # Save best model
        model_save_dir = UPLOAD_DIR / f"{request.dataset_id}_model"
        model_path = engine.save_best_model(str(model_save_dir))

        # Generate confusion matrix
        confusion_matrix_data = engine.generate_confusion_matrix()

        # Get summary
        summary = engine.get_summary()
        summary["model_path"] = str(model_save_dir)
        summary["dataset_id"] = request.dataset_id
        summary["confusion_matrix"] = confusion_matrix_data

        # Store engine for later use (confusion matrix, etc.)
        import pickle
        engine_path = model_save_dir / "engine.pkl"
        with open(engine_path, 'wb') as f:
            # Save only necessary data
            pickle.dump({
                'X_test': engine.X_test,
                'y_test': engine.y_test,
                'class_names': engine.class_names,
                'batch_size': engine.batch_size
            }, f)

        return {
            "status": "success",
            "problem_type": "image_classification",
            "preparation_info": prep_info,
            "summary": summary
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cv/datasets/{dataset_id}")
async def get_dataset_info(dataset_id: str):
    """
    Get information about uploaded dataset

    Args:
        dataset_id: ID of uploaded dataset

    Returns:
        Dataset information
    """
    try:
        dataset_path = UPLOAD_DIR / dataset_id

        if not dataset_path.exists():
            raise HTTPException(status_code=404, detail="Dataset not found")

        # Load dataset
        image_loader = ImageDataLoader(str(dataset_path))
        dataset = image_loader.load()

        return {
            "status": "success",
            "dataset_info": dataset["dataset_info"],
            "sample_count": 5
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cv/predict/{dataset_id}")
async def predict_image(dataset_id: str, file: UploadFile = File(...)):
    """
    Make prediction on a single image using trained model

    Args:
        dataset_id: ID of the dataset (used to find trained model)
        file: Image file to predict

    Returns:
        Top-3 predictions with class names and probabilities
    """
    try:
        model_dir = UPLOAD_DIR / f"{dataset_id}_model"

        if not model_dir.exists():
            raise HTTPException(
                status_code=404,
                detail="Trained model not found. Please train a model first."
            )

        # Load predictor
        predictor = CVPredictor(str(model_dir))

        # Read and preprocess image
        image_bytes = await file.read()
        pil_image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(pil_image)

        # Make prediction
        predictions = predictor.predict(image_array, top_k=3)

        # Get model info
        model_info = predictor.get_model_info()

        return {
            "status": "success",
            "predictions": predictions,
            "model_info": {
                "model_name": model_info['model_name'],
                "accuracy": model_info['metrics']['accuracy']
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cv/predict/gradcam/{dataset_id}")
async def predict_with_gradcam(dataset_id: str, file: UploadFile = File(...)):
    """
    Make prediction with Grad-CAM visualization

    Args:
        dataset_id: ID of the dataset (used to find trained model)
        file: Image file to predict

    Returns:
        Predictions with Grad-CAM heatmap and overlay
    """
    try:
        model_dir = UPLOAD_DIR / f"{dataset_id}_model"

        if not model_dir.exists():
            raise HTTPException(
                status_code=404,
                detail="Trained model not found. Please train a model first."
            )

        # Load predictor
        predictor = CVPredictor(str(model_dir))

        # Read and preprocess image
        image_bytes = await file.read()
        pil_image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(pil_image)

        # Make prediction with Grad-CAM
        result = predictor.visualize_prediction(image_array, top_k=3)

        # Get model info
        model_info = predictor.get_model_info()

        # Convert numpy arrays to base64 for JSON transmission
        if result.get('has_gradcam'):
            import base64
            from io import BytesIO
            from PIL import Image as PILImage

            # Convert overlay to base64
            overlay_array = np.array(result['overlay'], dtype=np.uint8)
            overlay_image = PILImage.fromarray(overlay_array)
            buffered = BytesIO()
            overlay_image.save(buffered, format="PNG")
            overlay_base64 = base64.b64encode(buffered.getvalue()).decode()

            result['overlay_base64'] = overlay_base64
            # Remove large arrays from response
            del result['overlay']
            del result['heatmap']

        return {
            "status": "success",
            "predictions": result['predictions'],
            "has_gradcam": result.get('has_gradcam', False),
            "overlay_base64": result.get('overlay_base64'),
            "error": result.get('error'),
            "model_info": {
                "model_name": model_info['model_name'],
                "accuracy": model_info['metrics']['accuracy']
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cv/models/{dataset_id}/download")
async def download_model(dataset_id: str):
    """
    Download trained model file

    Args:
        dataset_id: ID of the dataset

    Returns:
        Model file for download
    """
    try:
        from fastapi.responses import FileResponse

        model_dir = UPLOAD_DIR / f"{dataset_id}_model"
        model_file = model_dir / "best_model.pth"

        if not model_file.exists():
            raise HTTPException(status_code=404, detail="Model not found")

        return FileResponse(
            path=str(model_file),
            media_type='application/octet-stream',
            filename=f"cv_model_{dataset_id}.pth"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
