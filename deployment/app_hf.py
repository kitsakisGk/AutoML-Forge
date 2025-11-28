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
import cv2

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

st.info("🎯 **Try it:** Upload an image, select a model, and optionally enable Grad-CAM to see what the model focuses on!")

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

# Grad-CAM toggle
enable_gradcam = st.checkbox(
    "🎨 Enable Grad-CAM Visualization",
    value=False,
    help="Show heatmap of which image regions the model focuses on (interpretability)"
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
    image = Image.open(uploaded_file)

    if st.button("Predict", type="primary"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📷 Input Image")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("🔮 Predictions")

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
                        input_tensor = inputs["pixel_values"].to(device)
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

                        input_tensor = transform(image.convert('RGB')).unsqueeze(0).to(device)
                        inputs = {"pixel_values": input_tensor}

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

                    # Grad-CAM visualization
                    if enable_gradcam:
                        st.markdown("---")
                        st.markdown("### 🎨 Grad-CAM Visualization")

                        with st.spinner("Generating Grad-CAM heatmap..."):
                            try:
                                from pytorch_grad_cam import GradCAM
                                from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
                                from pytorch_grad_cam.utils.image import show_cam_on_image

                                # Get target layer
                                model_name = model_config["name"].lower()
                                if 'vit' in model_name:
                                    # For ViT, use the last layer norm
                                    target_layers = [model.vit.layernorm]
                                elif 'mobilenet' in model_name or 'efficientnet' in model_name or 'resnet' in model_name:
                                    # For CNNs, use the last convolutional layer
                                    target_layers = []
                                    for module in model.modules():
                                        if isinstance(module, nn.Conv2d):
                                            target_layers = [module]
                                else:
                                    target_layers = None

                                if target_layers:
                                    # Prepare image for Grad-CAM
                                    rgb_img = np.array(image.convert('RGB').resize((224, 224))) / 255.0

                                    # Create Grad-CAM
                                    cam = GradCAM(model=model, target_layers=target_layers)

                                    # Get the top prediction class
                                    target_class = top5_indices[0][0].item()
                                    targets = [ClassifierOutputTarget(target_class)]

                                    # Generate CAM
                                    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
                                    grayscale_cam = grayscale_cam[0, :]

                                    # Create visualization
                                    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

                                    # Display Grad-CAM
                                    st.image(visualization, caption=f"Grad-CAM for '{labels[target_class].title()}'", use_container_width=True)
                                    st.info("💡 **Tip**: Red/yellow regions show where the model focused most to make its prediction.")
                                else:
                                    st.warning("⚠️ Grad-CAM not available for this model architecture.")

                            except Exception as e:
                                st.warning(f"⚠️ Grad-CAM generation failed: {str(e)}")

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
- 🎨 **Grad-CAM**: Model interpretability with heatmaps ✅ **Available in this demo!**
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
