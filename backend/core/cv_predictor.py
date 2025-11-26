"""
Computer Vision Model Predictor
Loads trained models and makes predictions on new images
"""
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json


class CVPredictor:
    """Predictor for trained CV models"""

    def __init__(self, model_dir: str):
        """
        Initialize predictor

        Args:
            model_dir: Directory containing saved model
        """
        self.model_dir = Path(model_dir)
        self.model = None
        self.metadata = None
        self.device = None
        self.torch = None
        self.timm = None
        self.ViTForImageClassification = None

        self._import_dependencies()
        self._load_model()

    def _import_dependencies(self):
        """Import PyTorch and related libraries"""
        try:
            import torch
            import torch.nn as nn
            import timm
            from transformers import ViTForImageClassification

            self.torch = torch
            self.nn = nn
            self.timm = timm
            self.ViTForImageClassification = ViTForImageClassification
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        except ImportError as e:
            raise ImportError(
                f"Failed to import required libraries: {e}. "
                "Please install: pip install torch torchvision timm transformers"
            )

    def _load_model(self):
        """Load saved model from disk"""
        model_path = self.model_dir / "best_model.pth"
        metadata_path = self.model_dir / "model_metadata.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # Load model checkpoint
        checkpoint = self.torch.load(model_path, map_location=self.device)

        # Create model architecture
        model_config = checkpoint['model_config']
        model_type = model_config['type']
        model_name = model_config['name']
        num_classes = checkpoint['num_classes']

        if model_type == "vit":
            self.model = self.ViTForImageClassification.from_pretrained(
                model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        elif model_type == "timm":
            self.model = self.timm.create_model(
                model_name,
                pretrained=False,
                num_classes=num_classes
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

    def preprocess_image(self, image: np.ndarray) -> 'torch.Tensor':
        """
        Preprocess image for prediction

        Args:
            image: Numpy array (H, W, C) in range [0, 255] or [0, 1]

        Returns:
            Preprocessed tensor
        """
        # Ensure float32
        image = image.astype(np.float32)

        # Normalize to [0, 1] if needed
        if image.max() > 1.0:
            image = image / 255.0

        # Ensure RGB (3 channels)
        if len(image.shape) == 2:
            # Grayscale to RGB
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)

        # Resize to 224x224 (standard for ImageNet models)
        from PIL import Image
        pil_image = Image.fromarray((image * 255).astype(np.uint8))
        pil_image = pil_image.resize((224, 224))
        image = np.array(pil_image).astype(np.float32) / 255.0

        # Convert to tensor (C, H, W)
        tensor = self.torch.from_numpy(image).permute(2, 0, 1).float()

        # Add batch dimension
        tensor = tensor.unsqueeze(0)

        return tensor.to(self.device)

    def predict(self, image: np.ndarray, top_k: int = 3) -> List[Dict]:
        """
        Make prediction on a single image

        Args:
            image: Numpy array (H, W, C)
            top_k: Number of top predictions to return

        Returns:
            List of predictions with class names and probabilities
        """
        # Preprocess
        tensor = self.preprocess_image(image)

        # Predict
        with self.torch.no_grad():
            outputs = self.model(tensor)

            # Handle different output formats
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs

            # Get probabilities
            probs = self.torch.nn.functional.softmax(logits, dim=1)
            probs = probs.cpu().numpy()[0]

        # Get top-k predictions
        top_indices = np.argsort(probs)[::-1][:top_k]

        predictions = []
        for idx in top_indices:
            predictions.append({
                "class": self.metadata['class_names'][idx],
                "class_index": int(idx),
                "confidence": float(probs[idx])
            })

        return predictions

    def predict_batch(self, images: List[np.ndarray], top_k: int = 3) -> List[List[Dict]]:
        """
        Make predictions on multiple images

        Args:
            images: List of numpy arrays
            top_k: Number of top predictions per image

        Returns:
            List of prediction lists
        """
        results = []
        for image in images:
            predictions = self.predict(image, top_k=top_k)
            results.append(predictions)
        return results

    def get_model_info(self) -> Dict:
        """Get model metadata"""
        return {
            "model_name": self.metadata['model_name'],
            "num_classes": self.metadata['num_classes'],
            "class_names": self.metadata['class_names'],
            "metrics": self.metadata['metrics']
        }

    def get_gradcam_heatmap(self, image: np.ndarray, target_class: int = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for interpretability

        Args:
            image: Numpy array (H, W, C)
            target_class: Target class index (None = predicted class)

        Returns:
            Heatmap as numpy array (H, W) in range [0, 1]
        """
        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
            import torch.nn as nn

            # Preprocess image
            tensor = self.preprocess_image(image)

            # Determine target layer based on model type
            model_name = self.metadata['model_name'].lower()
            if 'vit' in model_name:
                # For ViT, use the last layer normalization
                target_layers = [self.model.vit.layernorm]
            elif 'resnet' in model_name or 'efficientnet' in model_name or 'mobilenet' in model_name:
                # For CNNs, use the last convolutional layer
                # Find the last Conv2d layer
                target_layers = []
                for module in self.model.modules():
                    if isinstance(module, nn.Conv2d):
                        target_layers = [module]
                if not target_layers:
                    raise ValueError("No Conv2d layers found in model")
            else:
                raise ValueError(f"Unsupported model for Grad-CAM: {model_name}")

            # Create GradCAM object
            cam = GradCAM(model=self.model, target_layers=target_layers)

            # Generate CAM
            if target_class is None:
                # Use predicted class
                targets = None
            else:
                targets = [ClassifierOutputTarget(target_class)]

            # Generate heatmap
            grayscale_cam = cam(input_tensor=tensor, targets=targets)

            # Return heatmap (first image in batch)
            return grayscale_cam[0]

        except ImportError:
            raise ImportError(
                "grad-cam library not installed. "
                "Please install: pip install grad-cam"
            )
        except Exception as e:
            raise RuntimeError(f"Error generating Grad-CAM: {str(e)}")

    def visualize_prediction(self, image: np.ndarray, top_k: int = 3) -> Dict:
        """
        Generate prediction with Grad-CAM visualization

        Args:
            image: Numpy array (H, W, C)
            top_k: Number of top predictions

        Returns:
            Dictionary with predictions and heatmap
        """
        # Get predictions
        predictions = self.predict(image, top_k=top_k)

        # Get Grad-CAM for top prediction
        try:
            top_class_idx = predictions[0]['class_index']
            heatmap = self.get_gradcam_heatmap(image, target_class=top_class_idx)

            # Overlay heatmap on original image
            from PIL import Image
            import cv2

            # Resize original image to 224x224 to match heatmap
            pil_image = Image.fromarray((image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8))
            pil_image = pil_image.resize((224, 224))
            image_resized = np.array(pil_image)

            # Convert heatmap to RGB
            heatmap_colored = cv2.applyColorMap(
                (heatmap * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

            # Overlay
            overlay = (0.6 * image_resized + 0.4 * heatmap_colored).astype(np.uint8)

            return {
                "predictions": predictions,
                "heatmap": heatmap.tolist(),
                "overlay": overlay.tolist(),
                "has_gradcam": True
            }

        except Exception as e:
            # Return predictions without Grad-CAM if it fails
            return {
                "predictions": predictions,
                "has_gradcam": False,
                "error": str(e)
            }
