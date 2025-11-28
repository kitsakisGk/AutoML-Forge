"""
AutoML Forge - Computer Vision Demo
Simplified standalone version for HuggingFace Spaces deployment
"""
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
import timm
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import numpy as np
import io
import zipfile
from pathlib import Path
import tempfile

st.set_page_config(
    page_title="AutoML Forge - CV Demo",
    page_icon="📸",
    layout="wide"
)

st.title("📸 AutoML Forge - Computer Vision Demo")
st.markdown("""
**State-of-the-art image classification** with Vision Transformers and CNNs.
This is a simplified demo version. For the full platform with AutoML training, see the [GitHub repository](https://github.com/kitsakisGk/AutoML-Forge).
""")

st.info("🎯 **Try it:** Upload an image and select a pre-trained model for instant predictions!")

# Model selection
model_choice = st.selectbox(
    "Choose a model",
    [
        "MobileNetV3 (Fast, mobile-optimized)",
        "ResNet18 (Lightweight CNN)",
        "EfficientNet-B0 (Balanced)",
        "ViT-Base (State-of-the-art Transformer)"
    ]
)

# Model mapping
MODEL_MAP = {
    "MobileNetV3 (Fast, mobile-optimized)": {"type": "timm", "name": "mobilenetv3_large_100"},
    "ResNet18 (Lightweight CNN)": {"type": "timm", "name": "resnet18"},
    "EfficientNet-B0 (Balanced)": {"type": "timm", "name": "efficientnet_b0"},
    "ViT-Base (State-of-the-art Transformer)": {"type": "vit", "name": "google/vit-base-patch16-224"}
}

# Image upload
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
    help="Upload any image for classification (ImageNet classes)"
)

if uploaded_file is not None:
    # Display image
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Input Image")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🔮 Predictions")

        if st.button("Predict", type="primary"):
            with st.spinner(f"Loading model and making predictions..."):
                try:
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                    # Load model
                    model_config = MODEL_MAP[model_choice]

                    if model_config["type"] == "vit":
                        model = ViTForImageClassification.from_pretrained(
                            model_config["name"]
                        )
                        processor = ViTImageProcessor.from_pretrained(model_config["name"])

                        # Preprocess
                        inputs = processor(images=image, return_tensors="pt")
                        inputs = {k: v.to(device) for k, v in inputs.items()}

                    else:  # timm models
                        model = timm.create_model(
                            model_config["name"],
                            pretrained=True
                        )

                        # Preprocess
                        transform = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]
                            )
                        ])

                        img_tensor = transform(image.convert('RGB')).unsqueeze(0).to(device)
                        inputs = {"pixel_values": img_tensor}

                    model = model.to(device)
                    model.eval()

                    # Predict
                    with torch.no_grad():
                        if model_config["type"] == "vit":
                            outputs = model(**inputs)
                            logits = outputs.logits
                        else:
                            logits = model(inputs["pixel_values"])

                        probs = torch.nn.functional.softmax(logits, dim=1)
                        top5_probs, top5_indices = torch.topk(probs, 5)

                    # Load ImageNet labels
                    import requests
                    imagenet_labels_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
                    labels = requests.get(imagenet_labels_url).json()

                    # Display predictions
                    st.markdown("### Top 5 Predictions")
                    for i, (prob, idx) in enumerate(zip(top5_probs[0], top5_indices[0]), 1):
                        confidence = prob.item()
                        label = labels[idx.item()]

                        color = "🟢" if i == 1 else "🟡" if i == 2 else "🟠" if i == 3 else "⚪"
                        st.markdown(f"{color} **{i}. {label.title()}** - {confidence:.2%}")
                        st.progress(confidence)

                    st.success(f"✅ Predicted using **{model_choice}** on {device}")

                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    import traceback
                    with st.expander("Error details"):
                        st.code(traceback.format_exc())

st.markdown("---")
st.markdown("""
### 🎓 About This Demo

This is a simplified demo of **AutoML Forge**, showcasing pre-trained models for image classification.

**Full Platform Features** (see [GitHub](https://github.com/kitsakisGk/AutoML-Forge)):
- 🤖 **Custom Training**: Train on your own datasets
- 📊 **AutoML**: Automatic model selection and training
- 🎨 **Grad-CAM**: Model interpretability with heatmaps
- 📈 **Confusion Matrix**: Per-class performance metrics
- 💾 **Model Export**: Download trained models
- 🔄 **MLflow Tracking**: Experiment versioning
- 📊 **Tabular ML**: Support for CSV/Excel data
- 🌍 **Bilingual**: English/German interface

**Models Used:**
- **MobileNetV3**: 5.4M params, optimized for mobile
- **ResNet18**: 11.7M params, classic CNN
- **EfficientNet-B0**: 5.3M params, balanced design
- **ViT-Base**: 86M params, state-of-the-art Transformer

All models pre-trained on **ImageNet** (1.2M images, 1000 classes).

---

**Author:** Kitsakis Giorgos
**GitHub:** [kitsakisGk/AutoML-Forge](https://github.com/kitsakisGk/AutoML-Forge)
**LinkedIn:** [georgios-kitsakis-gr](https://www.linkedin.com/in/georgios-kitsakis-gr/)
""")
