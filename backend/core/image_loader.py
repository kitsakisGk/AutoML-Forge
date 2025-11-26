"""
Image Data Loader - Load and preprocess image datasets
"""
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image


class ImageDataLoader:
    """
    Load and preprocess image datasets for Computer Vision tasks
    Supports:
    - Individual image files
    - Directory-based datasets (folder per class)
    - ZIP archives containing images
    """

    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    def __init__(self, data_path: str, image_size: Tuple[int, int] = (224, 224)):
        """
        Initialize Image Data Loader

        Args:
            data_path: Path to image file, directory, or ZIP archive
            image_size: Target size for resizing images (width, height)
        """
        self.data_path = Path(data_path)
        self.image_size = image_size
        self.images: List[np.ndarray] = []
        self.labels: List[str] = []
        self.class_names: List[str] = []
        self.num_classes: int = 0

    def load(self) -> Dict[str, any]:
        """
        Load images from the specified path

        Returns:
            Dictionary containing:
            - images: numpy array of images
            - labels: list of string labels
            - class_names: list of unique class names
            - num_classes: number of classes
            - dataset_info: metadata about the dataset
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Path not found: {self.data_path}")

        if self.data_path.is_file():
            # Single image file
            self._load_single_image()
        elif self.data_path.is_dir():
            # Directory with images (folder per class structure)
            self._load_directory()
        else:
            raise ValueError(f"Unsupported path type: {self.data_path}")

        # Get unique class names and assign numeric labels
        self.class_names = sorted(list(set(self.labels)))
        self.num_classes = len(self.class_names)

        # Convert string labels to numeric
        label_to_idx = {label: idx for idx, label in enumerate(self.class_names)}
        numeric_labels = [label_to_idx[label] for label in self.labels]

        dataset_info = {
            "total_images": len(self.images),
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "image_size": self.image_size,
            "class_distribution": self._get_class_distribution()
        }

        # Convert to numpy array with explicit dtype and shape validation
        images_array = np.array(self.images, dtype=np.float32)

        # Validate shape
        if len(images_array.shape) != 4:
            raise ValueError(
                f"Images array has wrong number of dimensions: {images_array.shape}. "
                f"Expected 4D array (N, H, W, C)"
            )

        if images_array.shape[3] != 3:
            raise ValueError(
                f"Images array has wrong number of channels: {images_array.shape[3]}. "
                f"Expected 3 channels (RGB)"
            )

        return {
            "images": images_array,
            "labels": np.array(numeric_labels, dtype=np.int64),
            "label_names": self.labels,  # Original string labels
            "class_names": self.class_names,
            "num_classes": self.num_classes,
            "dataset_info": dataset_info
        }

    def _load_single_image(self):
        """Load a single image file"""
        if self.data_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported image format: {self.data_path.suffix}")

        img = self._preprocess_image(str(self.data_path))
        self.images.append(img)
        self.labels.append("unknown")  # Single image, no label

    def _load_directory(self):
        """
        Load images from directory structure
        Expected structure:
        data_path/
            class1/
                img1.jpg
                img2.jpg
            class2/
                img1.jpg
                img2.jpg

        Also handles nested structure:
        data_path/
            dataset_name/
                class1/
                    img1.jpg
        """
        # Check if directory contains subdirectories (class folders)
        subdirs = [d for d in self.data_path.iterdir() if d.is_dir()]

        # Handle nested structure (e.g., ZIP extracted with root folder)
        # If there's only 1 subdirectory and it contains subdirectories, use that as root
        if len(subdirs) == 1:
            nested_subdirs = [d for d in subdirs[0].iterdir() if d.is_dir()]
            if nested_subdirs:
                # Use the nested level as the actual root
                self.data_path = subdirs[0]
                subdirs = nested_subdirs
                print(f"Detected nested structure, using: {self.data_path}")

        if subdirs:
            # Directory structure with class folders
            print(f"Found {len(subdirs)} classes: {[d.name for d in subdirs]}")

            for class_dir in subdirs:
                class_name = class_dir.name
                image_files = self._get_image_files(class_dir)

                if not image_files:
                    print(f"Warning: No images found in class '{class_name}'")
                    continue

                print(f"Loading {len(image_files)} images from class '{class_name}'")

                for img_path in image_files:
                    try:
                        img = self._preprocess_image(str(img_path))
                        self.images.append(img)
                        self.labels.append(class_name)
                    except Exception as e:
                        print(f"Warning: Failed to load {img_path}: {e}")
                        continue

            if not self.images:
                raise ValueError(
                    f"No images were loaded! Check that:\n"
                    f"1. ZIP file contains images in supported formats: {self.SUPPORTED_FORMATS}\n"
                    f"2. Images are organized in folders (one folder per class)\n"
                    f"3. Directory structure is: dataset.zip/class1/image1.jpg"
                )
        else:
            # Flat directory with images (all same class)
            image_files = self._get_image_files(self.data_path)
            for img_path in image_files:
                try:
                    img = self._preprocess_image(str(img_path))
                    self.images.append(img)
                    self.labels.append("class_0")  # Default class name
                except Exception as e:
                    print(f"Warning: Failed to load {img_path}: {e}")
                    continue

    def _get_image_files(self, directory: Path) -> List[Path]:
        """Get all image files in a directory"""
        image_files = []
        for ext in self.SUPPORTED_FORMATS:
            image_files.extend(directory.glob(f"*{ext}"))
            image_files.extend(directory.glob(f"*{ext.upper()}"))
        return image_files

    def _preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess a single image

        Args:
            image_path: Path to image file

        Returns:
            Preprocessed image as numpy array (H, W, C)
        """
        # Load image
        img = Image.open(image_path)

        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize
        img = img.resize(self.image_size, Image.Resampling.LANCZOS)

        # Convert to numpy array and normalize to [0, 1]
        img_array = np.array(img) / 255.0

        # Ensure 3D shape (H, W, C)
        if len(img_array.shape) == 2:
            # Grayscale image, add channel dimension
            img_array = np.stack([img_array] * 3, axis=-1)
        elif len(img_array.shape) != 3:
            raise ValueError(f"Unexpected image shape: {img_array.shape}")

        # Ensure float32 dtype
        img_array = img_array.astype(np.float32)

        return img_array

    def _get_class_distribution(self) -> Dict[str, int]:
        """Get count of images per class"""
        distribution = {}
        for label in self.labels:
            distribution[label] = distribution.get(label, 0) + 1
        return distribution

    def get_sample_images(self, num_samples: int = 5) -> List[Tuple[np.ndarray, str]]:
        """
        Get sample images with their labels

        Args:
            num_samples: Number of samples to return

        Returns:
            List of (image, label) tuples
        """
        indices = np.random.choice(len(self.images), min(num_samples, len(self.images)), replace=False)
        return [(self.images[i], self.labels[i]) for i in indices]
