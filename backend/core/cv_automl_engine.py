"""
Computer Vision AutoML Engine - Automatic image classification model training
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Note: PyTorch imports will be done dynamically to avoid import errors during installation
# import torch
# import torch.nn as nn
# from torchvision import transforms
# import timm
# from transformers import ViTForImageClassification, ViTImageProcessor


class CVAutoMLEngine:
    """
    Computer Vision AutoML Engine
    Trains multiple pre-trained models on image classification tasks using transfer learning

    Supported Models:
    - Vision Transformer (ViT) - State-of-the-art transformer-based model
    - ResNet50 - Classic CNN architecture
    - EfficientNet - Efficient and accurate CNN
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        class_names: List[str],
        test_size: float = 0.2,
        random_state: int = 42,
        num_epochs: int = 5,
        batch_size: int = 32,
        use_mlflow: bool = True,
        experiment_name: str = "CV_AutoML_Experiments"
    ):
        """
        Initialize CV AutoML Engine

        Args:
            images: Numpy array of images (N, H, W, C)
            labels: Numpy array of labels (N,)
            class_names: List of class names
            test_size: Proportion of test set
            random_state: Random seed
            num_epochs: Number of training epochs
            batch_size: Training batch size
            use_mlflow: Whether to track with MLflow
            experiment_name: MLflow experiment name
        """
        self.images = images
        self.labels = labels
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.test_size = test_size
        self.random_state = random_state
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.use_mlflow = use_mlflow
        self.experiment_name = experiment_name

        # Training data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # Model results
        self.models = {}
        self.results = {}

        # Import PyTorch here to avoid import errors
        self._import_dependencies()

    def _import_dependencies(self):
        """Import PyTorch and related libraries"""
        try:
            import torch
            import torch.nn as nn
            from torchvision import transforms
            import timm
            from transformers import ViTForImageClassification, ViTImageProcessor

            self.torch = torch
            self.nn = nn
            self.transforms = transforms
            self.timm = timm
            self.ViTForImageClassification = ViTForImageClassification
            self.ViTImageProcessor = ViTImageProcessor

            # Set device
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        except ImportError as e:
            raise ImportError(
                f"Failed to import required libraries: {e}. "
                "Please install: pip install torch torchvision timm transformers"
            )

    def prepare_data(self) -> Dict[str, Any]:
        """
        Prepare data for training (split, convert to tensors)

        Returns:
            Dictionary with preparation info
        """
        # Train-test split
        n_samples = len(self.images)
        n_test = int(n_samples * self.test_size)

        # Shuffle indices
        indices = np.random.RandomState(self.random_state).permutation(n_samples)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

        # Split data
        self.X_train = self.images[train_indices]
        self.X_test = self.images[test_indices]
        self.y_train = self.labels[train_indices]
        self.y_test = self.labels[test_indices]

        return {
            "n_samples": n_samples,
            "n_train": len(self.X_train),
            "n_test": len(self.X_test),
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "device": str(self.device)
        }

    def get_models(self) -> Dict[str, Any]:
        """
        Get pre-trained models for training
        All models available - trains fast ones first, then heavy ones

        Returns:
            Dictionary of model_name -> model config
        """
        return {
            # 3 Fast models + ViT (most important for CV expertise)
            "MobileNetV3": {
                "type": "timm",
                "name": "mobilenetv3_large_100",
                "description": "Fast and efficient CNN optimized for mobile devices",
                "speed": "fast"
            },
            "ResNet18": {
                "type": "timm",
                "name": "resnet18",
                "description": "Lightweight ResNet variant (4x faster than ResNet50)",
                "speed": "fast"
            },
            "EfficientNet-B0": {
                "type": "timm",
                "name": "efficientnet_b0",
                "description": "Efficient CNN with good accuracy/speed balance",
                "speed": "medium"
            },
            # State-of-the-art (important for portfolio)
            "ViT-Base": {
                "type": "vit",
                "name": "google/vit-base-patch16-224",
                "description": "Vision Transformer - State-of-the-art (slow on CPU but essential)",
                "speed": "slow"
            }
        }

    def train_models(self) -> Dict[str, Any]:
        """
        Train all models and collect results

        Returns:
            Dictionary with training results for all models
        """
        if self.use_mlflow:
            import mlflow
            mlflow.set_tracking_uri("file:./mlruns")
            mlflow.set_experiment(self.experiment_name)

        models_config = self.get_models()
        self.results = {}

        for model_name, config in models_config.items():
            print(f"\nTraining {model_name}...")

            if self.use_mlflow:
                import mlflow
                with mlflow.start_run(run_name=model_name):
                    result = self._train_single_model(model_name, config)
                    self.results[model_name] = result
                    # Store model object if training succeeded
                    if result.get('status') == 'success' and 'model' in result:
                        self.models[model_name] = result['model']
            else:
                result = self._train_single_model(model_name, config)
                self.results[model_name] = result
                # Store model object if training succeeded
                if result.get('status') == 'success' and 'model' in result:
                    self.models[model_name] = result['model']

        return self.results

    def _train_single_model(self, model_name: str, config: Dict) -> Dict[str, Any]:
        """
        Train a single model

        Args:
            model_name: Name of the model
            config: Model configuration

        Returns:
            Training results
        """
        try:
            # Log parameters
            if self.use_mlflow:
                import mlflow
                mlflow.log_param("model_type", model_name)
                mlflow.log_param("model_config", config["name"])
                mlflow.log_param("num_classes", self.num_classes)
                mlflow.log_param("num_epochs", self.num_epochs)
                mlflow.log_param("batch_size", self.batch_size)

            # Create model
            model = self._create_model(config)

            # Create data loaders
            train_loader, test_loader = self._create_data_loaders()

            # Training
            history = self._train_loop(model, train_loader)

            # Evaluation
            metrics = self._evaluate_model(model, test_loader)

            # Log metrics
            if self.use_mlflow:
                import mlflow
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)

            print(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}")

            return {
                "model": model,
                "metrics": metrics,
                "history": history,
                "status": "success"
            }

        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    def _create_model(self, config: Dict):
        """Create and initialize a model"""
        model_type = config["type"]
        model_name = config["name"]

        if model_type == "vit":
            # Vision Transformer from HuggingFace
            model = self.ViTForImageClassification.from_pretrained(
                model_name,
                num_labels=self.num_classes,
                ignore_mismatched_sizes=True
            )
        elif model_type == "timm":
            # Models from timm library
            model = self.timm.create_model(
                model_name,
                pretrained=True,
                num_classes=self.num_classes
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model = model.to(self.device)
        return model

    def _create_data_loaders(self):
        """Create PyTorch data loaders"""
        # Ensure images are numpy arrays with proper shape
        X_train = np.array(self.X_train, dtype=np.float32)
        X_test = np.array(self.X_test, dtype=np.float32)

        # Convert numpy arrays to tensors
        # Images are (N, H, W, C) in [0, 1], need to be (N, C, H, W)
        X_train_tensor = self.torch.from_numpy(X_train).permute(0, 3, 1, 2).float()
        X_test_tensor = self.torch.from_numpy(X_test).permute(0, 3, 1, 2).float()
        y_train_tensor = self.torch.from_numpy(self.y_train).long()
        y_test_tensor = self.torch.from_numpy(self.y_test).long()

        # Create datasets
        train_dataset = self.torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = self.torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)

        # Create data loaders
        train_loader = self.torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        test_loader = self.torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        return train_loader, test_loader

    def _train_loop(self, model, train_loader):
        """Training loop"""
        optimizer = self.torch.optim.Adam(model.parameters(), lr=0.0001)
        criterion = self.nn.CrossEntropyLoss()

        history = {"loss": [], "accuracy": []}

        for epoch in range(self.num_epochs):
            model.train()
            epoch_loss = 0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                optimizer.zero_grad()

                # Handle ViT output format
                if hasattr(model, 'config') and hasattr(model.config, 'model_type'):
                    outputs = model(images)
                    logits = outputs.logits
                else:
                    logits = model(images)

                loss = criterion(logits, labels)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Statistics
                epoch_loss += loss.item()
                _, predicted = self.torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = correct / total
            avg_loss = epoch_loss / len(train_loader)

            history["loss"].append(avg_loss)
            history["accuracy"].append(accuracy)

            print(f"  Epoch {epoch+1}/{self.num_epochs} - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}")

        return history

    def _evaluate_model(self, model, test_loader):
        """Evaluate model on test set"""
        model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []

        with self.torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Handle ViT output format
                if hasattr(model, 'config') and hasattr(model.config, 'model_type'):
                    outputs = model(images)
                    logits = outputs.logits
                else:
                    logits = model(images)

                _, predicted = self.torch.max(logits, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = correct / total

        # Calculate additional metrics
        from sklearn.metrics import precision_score, recall_score, f1_score

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision_score(all_labels, all_predictions, average='weighted', zero_division=0)),
            "recall": float(recall_score(all_labels, all_predictions, average='weighted', zero_division=0)),
            "f1_score": float(f1_score(all_labels, all_predictions, average='weighted', zero_division=0))
        }

        return metrics

    def get_best_model(self) -> Tuple[str, Dict[str, Any]]:
        """
        Get the best performing model

        Returns:
            Tuple of (model_name, model_results)
        """
        if not self.results:
            return None, None

        # Filter successful models
        successful_models = {
            name: result for name, result in self.results.items()
            if result.get('status') == 'success'
        }

        if not successful_models:
            return None, None

        # Get best based on accuracy
        best_model_name = max(
            successful_models.keys(),
            key=lambda name: successful_models[name]['metrics']['accuracy']
        )

        return best_model_name, successful_models[best_model_name]

    def save_best_model(self, save_dir: str) -> str:
        """
        Save the best trained model to disk

        Args:
            save_dir: Directory to save the model

        Returns:
            Path to saved model
        """
        import os
        best_model_name, best_model_info = self.get_best_model()

        if not best_model_info:
            raise ValueError("No trained models available to save")

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Get model
        model = self.models.get(best_model_name)
        if not model:
            raise ValueError(f"Model {best_model_name} not found")

        # Save model state dict and metadata
        model_path = os.path.join(save_dir, "best_model.pth")
        metadata_path = os.path.join(save_dir, "model_metadata.json")

        # Save model weights
        self.torch.save({
            'model_state_dict': model.state_dict(),
            'model_name': best_model_name,
            'model_config': self.get_models()[best_model_name],
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'metrics': best_model_info['metrics']
        }, model_path)

        # Save metadata as JSON
        import json
        with open(metadata_path, 'w') as f:
            json.dump({
                'model_name': best_model_name,
                'num_classes': self.num_classes,
                'class_names': self.class_names,
                'metrics': best_model_info['metrics']
            }, f, indent=2)

        return model_path

    def generate_confusion_matrix(self, model_name: str = None) -> Dict[str, Any]:
        """
        Generate confusion matrix for best or specified model

        Args:
            model_name: Name of model (None = best model)

        Returns:
            Dictionary with confusion matrix data
        """
        from sklearn.metrics import confusion_matrix

        if model_name is None:
            model_name, _ = self.get_best_model()

        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        model = self.models[model_name]
        model.eval()

        # Get predictions on test set
        X_test_tensor = self.torch.from_numpy(
            np.array(self.X_test, dtype=np.float32)
        ).permute(0, 3, 1, 2).float().to(self.device)

        y_pred = []
        with self.torch.no_grad():
            for i in range(0, len(X_test_tensor), self.batch_size):
                batch = X_test_tensor[i:i+self.batch_size]
                outputs = model(batch)

                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs

                predictions = self.torch.argmax(logits, dim=1)
                y_pred.extend(predictions.cpu().numpy())

        y_pred = np.array(y_pred)

        # Generate confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)

        # Per-class metrics
        per_class_metrics = []
        for i, class_name in enumerate(self.class_names):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            per_class_metrics.append({
                "class": class_name,
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "support": int(cm[i, :].sum())
            })

        return {
            "model_name": model_name,
            "confusion_matrix": cm.tolist(),
            "class_names": self.class_names,
            "per_class_metrics": per_class_metrics
        }

    def get_summary(self) -> Dict[str, Any]:
        """
        Get training summary

        Returns:
            Summary dictionary with all results
        """
        best_model_name, best_model_info = self.get_best_model()

        # Prepare results (exclude model objects for JSON serialization)
        results_summary = {}
        for name, result in self.results.items():
            if result.get('status') == 'success':
                results_summary[name] = {
                    "metrics": result['metrics'],
                    "final_train_accuracy": result['history']['accuracy'][-1] if result['history']['accuracy'] else 0
                }
            else:
                results_summary[name] = {
                    "status": "error",
                    "error": result.get('error')
                }

        return {
            "problem_type": "image_classification",
            "n_models_trained": len(self.results),
            "best_model": best_model_name,
            "best_model_score": best_model_info['metrics'] if best_model_info else None,
            "all_results": results_summary,
            "data_info": {
                "n_train": len(self.X_train) if self.X_train is not None else 0,
                "n_test": len(self.X_test) if self.X_test is not None else 0,
                "num_classes": self.num_classes,
                "class_names": self.class_names
            }
        }
