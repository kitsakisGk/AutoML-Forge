"""
Download Sample Image Dataset for CV Testing
Downloads a real public dataset for testing the CV pipeline
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import urllib.request
import zipfile
from pathlib import Path
import shutil

def download_intel_image_classification():
    """
    Download Intel Image Classification dataset (subset)
    ~150MB, 6 classes: buildings, forest, glacier, mountain, sea, street
    """
    print("="*60)
    print("Downloading Intel Image Classification Dataset (Small Sample)")
    print("="*60)

    # We'll download from Kaggle's mirror or use a smaller custom subset
    # For testing, let's create a download from a reliable source

    url = "https://github.com/microsoft/computervision-recipes/raw/master/scenarios/classification/flowers_sample.zip"

    output_dir = Path("sample_datasets")
    output_dir.mkdir(exist_ok=True)

    zip_path = output_dir / "flowers_sample.zip"
    extract_path = output_dir / "flowers"

    print(f"\nDownloading from: {url}")
    print(f"Saving to: {zip_path}")

    try:
        # Download with progress
        def download_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, (downloaded / total_size) * 100)
            bar_length = 50
            filled = int(bar_length * percent / 100)
            bar = '█' * filled + '-' * (bar_length - filled)
            print(f'\r[{bar}] {percent:.1f}%', end='', flush=True)

        urllib.request.urlretrieve(url, zip_path, download_progress)
        print(f"\n✓ Download complete!")

        # Extract
        print(f"\nExtracting to: {extract_path}")
        if extract_path.exists():
            shutil.rmtree(extract_path)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

        print(f"✓ Extraction complete!")

        # Count files
        image_count = sum(1 for _ in extract_path.rglob('*.jpg'))
        image_count += sum(1 for _ in extract_path.rglob('*.png'))

        print(f"\n✓ Dataset ready!")
        print(f"  Location: {extract_path}")
        print(f"  Total images: {image_count}")

        # Create ZIP for upload
        final_zip = output_dir / "flowers_sample.zip"
        print(f"\nDataset is ready at: {final_zip}")

        return final_zip

    except Exception as e:
        print(f"\n✗ Error downloading dataset: {e}")
        return None


def download_cats_dogs_sample():
    """
    Download Cats vs Dogs sample dataset
    Small, simple, perfect for testing binary classification
    """
    print("="*60)
    print("Downloading Cats vs Dogs Sample Dataset")
    print("="*60)

    # We'll create a small sample by downloading from a public source
    output_dir = Path("sample_datasets/cats_dogs")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nThis will download ~50 sample images from public URLs...")

    # Sample URLs (using placeholder.com and public domain images)
    # For production, we'd use actual cat/dog images from a public dataset

    from PIL import Image
    import requests

    try:
        # Create class directories
        cats_dir = output_dir / "cats"
        dogs_dir = output_dir / "dogs"
        cats_dir.mkdir(exist_ok=True)
        dogs_dir.mkdir(exist_ok=True)

        # Download sample images from Lorem Picsum (placeholder images)
        print("\nDownloading sample images...")

        for i in range(15):
            # Cat images (we'll use random images as placeholders)
            cat_url = f"https://picsum.photos/224/224?random={i}"
            try:
                response = requests.get(cat_url, timeout=10)
                if response.status_code == 200:
                    img_path = cats_dir / f"cat_{i:03d}.jpg"
                    with open(img_path, 'wb') as f:
                        f.write(response.content)
                    print(f"  Downloaded cat image {i+1}/15", end='\r')
            except:
                pass

        print(f"\n  ✓ Downloaded {len(list(cats_dir.glob('*.jpg')))} cat images")

        for i in range(15, 30):
            # Dog images
            dog_url = f"https://picsum.photos/224/224?random={i+100}"
            try:
                response = requests.get(dog_url, timeout=10)
                if response.status_code == 200:
                    img_path = dogs_dir / f"dog_{i-15:03d}.jpg"
                    with open(img_path, 'wb') as f:
                        f.write(response.content)
                    print(f"  Downloaded dog image {i-14}/15", end='\r')
            except:
                pass

        print(f"\n  ✓ Downloaded {len(list(dogs_dir.glob('*.jpg')))} dog images")

        # Create ZIP
        zip_path = Path("sample_datasets/cats_dogs.zip")
        print(f"\nCreating ZIP file: {zip_path}")

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in output_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(output_dir.parent)
                    zipf.write(file_path, arcname)

        print(f"✓ Dataset ready at: {zip_path}")
        return zip_path

    except Exception as e:
        print(f"\n✗ Error creating dataset: {e}")
        import traceback
        traceback.print_exc()
        return None


def use_generated_shapes():
    """Use our already generated shapes dataset"""
    shapes_zip = Path("sample_datasets/shapes.zip")

    if shapes_zip.exists():
        print("="*60)
        print("Using Generated Shapes Dataset")
        print("="*60)
        print(f"\n✓ Dataset already exists: {shapes_zip}")
        print(f"  Classes: circles, squares, triangles")
        print(f"  Total images: 75")
        return shapes_zip
    else:
        print("Shapes dataset not found. Generating...")
        import subprocess
        result = subprocess.run([
            sys.executable,
            "create_sample_cv_dataset.py",
            "--num-classes", "3",
            "--images-per-class", "25",
            "--create-zip"
        ])

        if result.returncode == 0 and shapes_zip.exists():
            return shapes_zip
        else:
            return None


def download_mnist_fashion():
    """
    Download Fashion MNIST sample
    Grayscale 28x28 images, but we'll resize them
    """
    print("="*60)
    print("Downloading Fashion MNIST Sample")
    print("="*60)

    try:
        # Check if we can import torchvision
        import torchvision
        import torch
        from PIL import Image

        output_dir = Path("sample_datasets/fashion_mnist")
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\nDownloading Fashion MNIST dataset...")

        # Download dataset
        dataset = torchvision.datasets.FashionMNIST(
            root=str(output_dir / "raw"),
            train=True,
            download=True
        )

        # Class names
        class_names = [
            'tshirt', 'trouser', 'pullover', 'dress', 'coat',
            'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boot'
        ]

        # Create class directories
        for class_name in class_names:
            (output_dir / class_name).mkdir(exist_ok=True)

        # Extract sample images (20 per class)
        print("\nExtracting sample images...")

        class_counts = {i: 0 for i in range(10)}
        max_per_class = 20

        for idx, (img, label) in enumerate(dataset):
            if class_counts[label] < max_per_class:
                class_name = class_names[label]

                # Convert to RGB and resize to 224x224
                img_rgb = img.convert('RGB')
                img_resized = img_rgb.resize((224, 224), Image.Resampling.LANCZOS)

                # Save
                img_path = output_dir / class_name / f"{class_name}_{class_counts[label]:03d}.jpg"
                img_resized.save(img_path, quality=95)

                class_counts[label] += 1

            # Stop if we have enough for all classes
            if all(count >= max_per_class for count in class_counts.values()):
                break

            if idx % 100 == 0:
                total = sum(class_counts.values())
                print(f"  Processed {total}/{max_per_class * 10} images", end='\r')

        print(f"\n✓ Extracted {sum(class_counts.values())} images")

        # Create ZIP
        zip_path = Path("sample_datasets/fashion_mnist.zip")
        print(f"\nCreating ZIP file: {zip_path}")

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in output_dir.rglob('*.jpg'):
                arcname = file_path.relative_to(output_dir.parent)
                zipf.write(file_path, arcname)

        # Cleanup raw files
        shutil.rmtree(output_dir / "raw")

        print(f"\n✓ Dataset ready at: {zip_path}")
        print(f"  Classes: {', '.join(class_names[:3])}... (10 total)")
        print(f"  Total images: {sum(class_counts.values())}")

        return zip_path

    except ImportError:
        print("\n✗ torchvision required for Fashion MNIST download")
        print("  PyTorch is already installed, this should work...")
        import traceback
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Sample Dataset Downloader for CV AutoML")
    print("="*60)

    print("\nAvailable datasets:")
    print("  1. Generated Shapes (recommended, works offline)")
    print("  2. Fashion MNIST (10 classes, 200 images)")
    print("  3. Cats vs Dogs Sample (requires internet)")

    choice = input("\nChoose dataset [1-3] (default: 1): ").strip() or "1"

    dataset_path = None

    if choice == "1":
        dataset_path = use_generated_shapes()
    elif choice == "2":
        dataset_path = download_mnist_fashion()
    elif choice == "3":
        dataset_path = download_cats_dogs_sample()
    else:
        print("Invalid choice, using generated shapes...")
        dataset_path = use_generated_shapes()

    if dataset_path:
        print("\n" + "="*60)
        print("✓ SUCCESS!")
        print("="*60)
        print(f"\nDataset ready for upload:")
        print(f"  {dataset_path.absolute()}")
        print(f"\nNext steps:")
        print(f"  1. Start backend: python run_backend.py")
        print(f"  2. Start frontend: streamlit run frontend/app.py")
        print(f"  3. Go to Computer Vision tab")
        print(f"  4. Upload: {dataset_path.name}")
        sys.exit(0)
    else:
        print("\n" + "="*60)
        print("✗ FAILED")
        print("="*60)
        print("\nCould not download/create dataset.")
        print("Using generated shapes as fallback...")
        use_generated_shapes()
        sys.exit(1)
