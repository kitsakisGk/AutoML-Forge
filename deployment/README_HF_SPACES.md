# 🚀 Deploy to HuggingFace Spaces

This guide shows how to deploy the AutoML Forge Computer Vision demo to HuggingFace Spaces.

## 📋 Prerequisites

- HuggingFace account (free): https://huggingface.co/join
- Git installed locally

## 🎯 Quick Deploy (5 minutes)

### Step 1: Create a Space

1. Go to https://huggingface.co/spaces
2. Click **"Create new Space"**
3. Configure:
   - **Space name**: `automl-forge-cv-demo` (or your choice)
   - **License**: MIT
   - **Space SDK**: **Streamlit**
   - **Space hardware**: CPU basic (free)
   - **Visibility**: Public

### Step 2: Clone Your Space

```bash
# Clone the empty space
git clone https://huggingface.co/spaces/YOUR_USERNAME/automl-forge-cv-demo
cd automl-forge-cv-demo
```

### Step 3: Copy Demo Files

From your AutoML-Forge project, copy these files to the space directory:

```bash
# Copy demo app (rename to app.py for HF Spaces)
cp /path/to/AutoML-Forge/app_hf.py app.py

# Copy requirements
cp /path/to/AutoML-Forge/requirements_hf.txt requirements.txt
```

### Step 4: Push to HuggingFace

```bash
# Add files
git add app.py requirements.txt

# Commit
git commit -m "Initial deployment: AutoML Forge CV Demo"

# Push to HuggingFace
git push
```

### Step 5: Wait for Build

- HuggingFace will automatically build your Space (~3-5 minutes)
- You'll see the build logs in the Space interface
- Once done, your app will be live! 🎉

## 🌐 Your Live Demo

Your demo will be available at:
```
https://huggingface.co/spaces/YOUR_USERNAME/automl-forge-cv-demo
```

Share this link with recruiters, add it to your resume, or embed it in your portfolio!

## 📝 Optional: Customize the Space

### Add a Custom README

Create a `README.md` in your space directory:

```markdown
---
title: AutoML Forge - CV Demo
emoji: 📸
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
---

# AutoML Forge - Computer Vision Demo

State-of-the-art image classification with Vision Transformers and CNNs.

**Features:**
- 4 pre-trained models (MobileNetV3, ResNet18, EfficientNet, ViT)
- Real-time predictions on ImageNet classes
- Optimized for fast inference

**Full Project:** [GitHub](https://github.com/kitsakisGk/AutoML-Forge)
```

### Upgrade to GPU (Optional)

For faster predictions (especially with ViT):
1. Go to your Space settings
2. Upgrade to "GPU basic" (free tier available)
3. Predictions will be 10-20x faster!

## 🎨 What Users Will See

The demo showcases:
- ✅ **4 State-of-the-art Models**: MobileNetV3, ResNet18, EfficientNet-B0, ViT-Base
- ✅ **Real-time Predictions**: Upload image → get instant results
- ✅ **Top-5 Predictions**: With confidence scores
- ✅ **Production-Ready**: Deployed and working live

## 📊 Alternative: Full Platform Deployment

For deploying the **full AutoML platform** (with training, Grad-CAM, etc.):

### Option 1: Railway.app (Recommended)
- Supports both frontend and backend
- Free tier available
- Easy GitHub integration

### Option 2: Render.com
- Free tier for web services
- Supports Docker (if you add Dockerfiles later)

### Option 3: Vercel (Frontend only)
- Deploy Streamlit frontend
- Backend needs separate hosting

## 🤝 Need Help?

- HuggingFace Spaces Docs: https://huggingface.co/docs/hub/spaces
- Issues: https://github.com/kitsakisGk/AutoML-Forge/issues

---

**Author:** Kitsakis Giorgos
**GitHub:** [kitsakisGk](https://github.com/kitsakisGk)
