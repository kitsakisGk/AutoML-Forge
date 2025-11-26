# 📸 Computer Vision AutoML - Quick Start Guide

Welcome to the Computer Vision module of AutoML Pipeline Builder! This guide will help you train state-of-the-art image classification models in minutes.

---

## 🎯 What You Get

Train and compare **4 state-of-the-art models** automatically:

1. **MobileNetV3** - Super fast, mobile-optimized (~1-2 min)
2. **ResNet18** - Lightweight CNN, fast training (~1-2 min)
3. **EfficientNet-B0** - Balanced speed/accuracy (~2-3 min)
4. **Vision Transformer (ViT)** - State-of-the-art from Google (~5-8 min)

All models use **transfer learning** with ImageNet pre-trained weights.

**Total training time on slow PC: ~10-12 minutes** (fast models finish first!)

---

## 🚀 Quick Start (5 Minutes)

### Step 1: Start the Application

```bash
# Terminal 1 - Start Backend
python run_backend.py

# Terminal 2 - Start Frontend
streamlit run frontend/app.py
```

### Step 2: Prepare Your Dataset

Create a ZIP file with this structure:

```
dataset.zip
    /class1
        image1.jpg
        image2.jpg
        ...
    /class2
        image1.jpg
        image2.jpg
        ...
    /class3
        ...
```

**Requirements:**
- Minimum 2 classes
- At least 5-10 images per class (50-100+ recommended)
- Supported formats: JPG, JPEG, PNG, BMP, TIFF, WEBP
- Folder names = Class names

### Step 3: Upload & Train

1. Open http://localhost:8501 in your browser
2. Click **"Computer Vision"** in the sidebar
3. Go to **"Upload Dataset"** tab
4. Upload your ZIP file
5. Go to **"Train Models"** tab
6. Configure training parameters (or use defaults)
7. Click **"Start Training"**
8. Go to **"Results"** tab to see model comparison

---

## 🧪 Test with Sample Dataset

We've included a script to generate sample datasets for testing:

```bash
# Create sample dataset with 3 classes (circles, squares, triangles)
python create_sample_cv_dataset.py --num-classes 3 --images-per-class 25 --create-zip

# This creates: sample_datasets/shapes.zip
```

Then upload `sample_datasets/shapes.zip` to the CV interface!

---

## ⚙️ Training Configuration

### Number of Epochs (1-20)
- **Fewer epochs (1-5)**: Faster training, good for testing
- **More epochs (10-20)**: Better accuracy, takes longer
- **Recommended**: 5 epochs for small datasets, 10-15 for larger datasets

### Batch Size (8, 16, 32, 64)
- **Smaller (8-16)**: Less memory, slower training
- **Larger (32-64)**: More memory, faster training
- **Recommended**: 32 (balanced)

### Test Set Size (0.1-0.3)
- Proportion of data to use for testing
- **Recommended**: 0.2 (20% for testing)

---

## 📊 Understanding Results

After training, you'll see:

### Model Comparison Table
- **Accuracy**: Overall classification accuracy
- **Precision**: How many predicted positives are correct
- **Recall**: How many actual positives are found
- **F1 Score**: Harmonic mean of precision and recall

### Best Model
- Automatically selected based on highest test accuracy
- Highlighted in green in the results table

---

## 💡 Tips for Best Results

### Data Quality
✅ **DO:**
- Use clear, high-quality images
- Balance classes (similar number of images per class)
- Use 50-100+ images per class
- Include variety in your images (different angles, lighting, etc.)

❌ **DON'T:**
- Mix unrelated images in the same class
- Use extremely low-resolution images
- Have huge class imbalance (e.g., 100 images in class A, 5 in class B)

### Training Settings
- Start with **5 epochs** and **batch size 32**
- If accuracy is low, try **more epochs** (10-15)
- If you have limited data, use **smaller batch sizes** (16 or 8)

### Model Selection
- **MobileNetV3**: Best for mobile/edge deployment, super fast
- **ResNet18**: Reliable lightweight baseline, fast training
- **EfficientNet-B0**: Balanced accuracy and speed
- **ViT**: Best accuracy, state-of-the-art (most important for CV expertise!)

---

## 🔧 Troubleshooting

### "Dataset not found" error
- Make sure you uploaded the dataset successfully
- Check that the dataset ID is showing in the Train tab

### Training takes too long
- Reduce number of epochs (try 1-2 for quick testing)
- Reduce number of images in your dataset
- Use smaller batch size

### Low accuracy
- Add more images per class (aim for 50-100+)
- Balance your classes
- Increase number of epochs
- Check for mislabeled images

### Out of memory error
- Reduce batch size (try 16 or 8)
- Reduce image resolution in your dataset
- Close other applications

---

## 🎓 What's Happening Under the Hood?

### Transfer Learning
Instead of training from scratch, we use models pre-trained on ImageNet (1.2M images, 1000 classes). We:

1. Load pre-trained model
2. Replace final classification layer with new one (for your classes)
3. Fine-tune the entire model on your data

This gives you:
- **Better accuracy** with less data
- **Faster training** (5-10 minutes vs hours/days)
- **State-of-the-art performance**

### MLflow Tracking
All experiments are automatically tracked with MLflow:
- Model parameters
- Training metrics
- Model artifacts
- Cross-validation scores

View experiments:
```bash
python view_mlflow_results.py
```

---

## 📚 Next Steps

### Week 1 (Core CV Features) ✅
- [x] Image data loader
- [x] Vision Transformer (ViT) integration
- [x] ResNet50 model
- [x] EfficientNet model
- [x] CV AutoML engine
- [x] API endpoints
- [x] Streamlit interface
- [ ] Grad-CAM visualization (coming soon)
- [ ] End-to-end testing

### Week 2 (Production Features)
- [ ] Model deployment guide
- [ ] API endpoint for predictions
- [ ] Architecture diagram
- [ ] Performance benchmarks
- [ ] Update README with CV screenshots

---

## 🤝 Contributing

Found a bug or have a feature request? Please create an issue on GitHub!

---

## 📄 License

MIT License - See LICENSE file for details

---

## 👤 Author

**Kitsakis Giorgos**
- LinkedIn: [georgios-kitsakis-gr](https://www.linkedin.com/in/georgios-kitsakis-gr/)
- GitHub: [kitsakisGk](https://github.com/kitsakisGk)

---

**Happy Training! 🚀**
