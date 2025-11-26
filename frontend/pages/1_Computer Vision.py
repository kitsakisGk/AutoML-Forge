"""
AutoML Forge - Computer Vision
"""
import streamlit as st
import requests
import zipfile
import io
from pathlib import Path
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="AutoML Forge - Computer Vision",
    page_icon="📸",
    layout="wide"
)

# API configuration
API_BASE_URL = "http://localhost:8000/api"

# Initialize session state
if "cv_dataset_id" not in st.session_state:
    st.session_state.cv_dataset_id = None

if "cv_training_results" not in st.session_state:
    st.session_state.cv_training_results = None

# Main content
st.title("📸 Computer Vision AutoML")
st.markdown("""
Train state-of-the-art image classification models automatically with **4 models**:
MobileNetV3, ResNet18, EfficientNet-B0, and Vision Transformers (ViT).
Just upload your image dataset and let AutoML do the rest!
""")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload Dataset", "🤖 Train Models", "📊 Results", "🔮 Predictions"])

# Tab 1: Upload Dataset
with tab1:
    st.header("Upload Image Dataset")

    st.markdown("""
    ### Dataset Requirements

    Upload a **ZIP file** containing your image dataset with the following structure:

    ```
    dataset.zip
        /class1
            image1.jpg
            image2.png
            ...
        /class2
            image1.jpg
            image2.png
            ...
        /class3
            ...
    ```

    - **Folder names** = Class names
    - **Supported formats**: JPG, JPEG, PNG, BMP, TIFF, WEBP
    - **Minimum**: 2 classes, at least 5 images per class
    - **Recommended**: 50-100+ images per class for best results
    """)

    uploaded_zip = st.file_uploader(
        "Choose a ZIP file containing your image dataset",
        type=["zip"],
        help="ZIP file with folder-per-class structure"
    )

    if uploaded_zip is not None:
        # Show upload button
        if st.button("🚀 Upload Dataset", type="primary"):
            with st.spinner("Uploading and processing dataset..."):
                try:
                    # Upload to API
                    files = {"file": (uploaded_zip.name, uploaded_zip.getvalue())}
                    response = requests.post(
                        f"{API_BASE_URL}/cv/upload",
                        files=files,
                        timeout=120
                    )

                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.cv_dataset_id = data["dataset_id"]
                        dataset_info = data["dataset_info"]

                        st.success("✅ Dataset uploaded successfully!")

                        # Display dataset info
                        st.subheader("Dataset Information")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Images", dataset_info["total_images"])
                        with col2:
                            st.metric("Number of Classes", dataset_info["num_classes"])
                        with col3:
                            st.metric("Image Size", f"{dataset_info['image_size'][0]}x{dataset_info['image_size'][1]}")

                        # Show class distribution
                        st.subheader("Class Distribution")
                        class_counts = dataset_info["class_distribution"]

                        # Create DataFrame for better display
                        import pandas as pd
                        df = pd.DataFrame([
                            {"Class": class_name, "Images": count}
                            for class_name, count in class_counts.items()
                        ])

                        # Display table and chart
                        st.dataframe(df, width='stretch', hide_index=True)

                        # Create proper bar chart
                        chart_data = pd.DataFrame({
                            "Images": list(class_counts.values())
                        }, index=list(class_counts.keys()))

                        st.bar_chart(chart_data)

                        st.info("📝 Dataset ID: `{}`".format(st.session_state.cv_dataset_id))
                        st.success("👉 Go to the **Train Models** tab to start training!")

                    else:
                        st.error(f"Upload failed: {response.text}")

                except Exception as e:
                    st.error(f"Error uploading dataset: {str(e)}")

# Tab 2: Train Models
with tab2:
    st.header("Train Image Classification Models")

    if st.session_state.cv_dataset_id is None:
        st.warning("⚠️ Please upload a dataset first in the **Upload Dataset** tab.")
    else:
        st.success(f"✅ Dataset loaded: `{st.session_state.cv_dataset_id}`")

        # Get dataset info
        try:
            response = requests.get(
                f"{API_BASE_URL}/cv/datasets/{st.session_state.cv_dataset_id}",
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                if "dataset_info" in data:
                    dataset_info = data["dataset_info"]

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Images", dataset_info.get("total_images", 0))
                    with col2:
                        st.metric("Classes", dataset_info.get("num_classes", 0))
                    with col3:
                        img_size = dataset_info.get("image_size", [224, 224])
                        st.metric("Size", f"{img_size[0]}x{img_size[1]}")
                else:
                    st.warning("Dataset info not available")
        except Exception as e:
            st.error(f"Error loading dataset info: {str(e)}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())

        st.markdown("---")

        # Training configuration
        st.subheader("Training Configuration")

        col1, col2, col3 = st.columns(3)

        with col1:
            num_epochs = st.slider(
                "Number of Epochs",
                min_value=1,
                max_value=20,
                value=5,
                help="More epochs = longer training but potentially better results"
            )

        with col2:
            batch_size = st.select_slider(
                "Batch Size",
                options=[8, 16, 32, 64],
                value=32,
                help="Larger batch = faster training but more memory usage"
            )

        with col3:
            test_size = st.slider(
                "Test Set Size",
                min_value=0.1,
                max_value=0.3,
                value=0.2,
                step=0.05,
                help="Proportion of data to use for testing"
            )

        st.markdown("---")

        # Model info
        st.subheader("4 Models to Train (Ordered by Speed)")

        st.info("💡 **Tip**: 3 fast models finish in ~3-5 min, then ViT takes ~5-8 min more. Total: ~10-12 min on slow PC")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **⚡ Fast Models** (~1-2 min each)
            - **MobileNetV3**: Mobile-optimized, super fast
            - **ResNet18**: Lightweight CNN, fast training
            - **EfficientNet-B0**: Balanced speed/accuracy
            """)

        with col2:
            st.markdown("""
            **🎯 State-of-the-Art** (~5-8 min)
            - **ViT-Base**: Vision Transformer (most important for CV expertise!)

            *ViT trains last, so you get fast results first, then the best model!*
            """)

        st.markdown("---")

        # Start training button
        if st.button("🚀 Start Training", type="primary", width='stretch'):
            with st.spinner("Training models... This may take several minutes ⏳"):
                try:
                    # Send training request
                    payload = {
                        "dataset_id": st.session_state.cv_dataset_id,
                        "num_epochs": num_epochs,
                        "batch_size": batch_size,
                        "test_size": test_size
                    }

                    response = requests.post(
                        f"{API_BASE_URL}/cv/train/start",
                        json=payload,
                        timeout=1200  # 20 minutes timeout for slow PC
                    )

                    if response.status_code == 200:
                        results = response.json()
                        st.session_state.cv_training_results = results

                        st.success("✅ Training completed successfully!")
                        st.balloons()

                        # Show quick summary
                        summary = results.get("summary", {})
                        if summary.get("best_model"):
                            st.info(f"🏆 Best Model: **{summary['best_model']}**")

                        st.success("📊 Go to the **Results** tab to see detailed model comparison!")

                    else:
                        st.error(f"Training failed: {response.text}")
                        with st.expander("Response details"):
                            st.json(response.json() if response.headers.get('content-type') == 'application/json' else response.text)

                except requests.exceptions.Timeout:
                    st.error("⏱️ Training timeout. The training is taking too long. Try reducing epochs or dataset size.")
                except Exception as e:
                    st.error(f"❌ Error during training: {str(e)}")
                    import traceback
                    with st.expander("Error details"):
                        st.code(traceback.format_exc())

# Tab 3: Results
with tab3:
    st.header("Training Results")

    if st.session_state.cv_training_results is None:
        st.warning("⚠️ No training results yet. Train models first in the **Train Models** tab.")
    else:
        results = st.session_state.cv_training_results
        summary = results.get("summary", {})

        # Check if we have a best model
        if summary.get('best_model'):
            st.success(f"🏆 Best Model: **{summary['best_model']}**")
        else:
            st.error("❌ All models failed to train. Please check the error messages below.")

        # Display metrics for all models
        st.subheader("Model Comparison")

        import pandas as pd

        # Create comparison DataFrame
        comparison_data = []
        error_messages = []

        for model_name, model_results in summary.get("all_results", {}).items():
            if model_results.get("status") == "error":
                error_messages.append(f"**{model_name}**: {model_results.get('error', 'Unknown error')}")
            else:
                metrics = model_results.get("metrics", {})
                comparison_data.append({
                    "Model": model_name,
                    "Accuracy": f"{metrics.get('accuracy', 0):.4f}",
                    "Precision": f"{metrics.get('precision', 0):.4f}",
                    "Recall": f"{metrics.get('recall', 0):.4f}",
                    "F1 Score": f"{metrics.get('f1_score', 0):.4f}",
                    "Train Accuracy": f"{model_results.get('final_train_accuracy', 0):.4f}"
                })

        # Show errors if any
        if error_messages:
            st.error("### Training Errors")
            for msg in error_messages:
                st.markdown(msg)

        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)

            # Style the best model row
            def highlight_best(row):
                if summary.get('best_model') and row["Model"] == summary["best_model"]:
                    return ['background-color: #90EE90'] * len(row)
                return [''] * len(row)

            st.dataframe(
                df_comparison.style.apply(highlight_best, axis=1),
                width='stretch',
                hide_index=True
            )
        else:
            st.warning("No successful model training results to display.")

        st.markdown("---")

        # Detailed metrics for each model
        st.subheader("Detailed Model Metrics")

        for model_name, model_results in summary["all_results"].items():
            if model_results.get("status") != "error":
                with st.expander(f"📊 {model_name} {'🏆' if model_name == summary['best_model'] else ''}"):
                    metrics = model_results["metrics"]

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                    with col2:
                        st.metric("Precision", f"{metrics['precision']:.4f}")
                    with col3:
                        st.metric("Recall", f"{metrics['recall']:.4f}")
                    with col4:
                        st.metric("F1 Score", f"{metrics['f1_score']:.4f}")

                    st.markdown(f"**Final Training Accuracy**: {model_results['final_train_accuracy']:.4f}")

        st.markdown("---")

        # Data info
        st.subheader("Dataset Information")
        data_info = summary["data_info"]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training Samples", data_info["n_train"])
        with col2:
            st.metric("Test Samples", data_info["n_test"])
        with col3:
            st.metric("Number of Classes", data_info["num_classes"])

        st.markdown(f"**Classes**: {', '.join(data_info['class_names'])}")

# Sidebar
with st.sidebar:
    st.title("ℹ️ About CV AutoML")

    st.markdown("""
    ### What is this?

    Automatic image classification using **transfer learning** with state-of-the-art pre-trained models.

    ### 4 Models (Fast → Slow)

    **⚡ Fast** (~1-2 min each):
    - MobileNetV3, ResNet18, EfficientNet-B0

    **🎯 State-of-the-Art** (~5-8 min):
    - ViT-Base (Vision Transformer)

    ### How it works

    1. Upload ZIP with images
    2. Configure training (2 epochs recommended)
    3. Train 4 models (fast ones first!)
    4. Stop anytime or wait for all
    5. Compare & use the best!

    ### Tips for Best Results

    - Use 20-50+ images per class
    - Balance classes (equal images per class)
    - Start with 2 epochs, batch size 16
    - Fast models finish in ~5 min
    - All 4 models take ~10-12 min total
    """)

    st.markdown("---")

    st.markdown("""
    ### Powered By

    - PyTorch
    - HuggingFace Transformers
    - timm (PyTorch Image Models)
    - MLflow
    """)

# Tab 4: Predictions
with tab4:
    st.header("Make Predictions")
    st.markdown("""
    Upload an image to get predictions from your trained model. The model will show:
    - Top 3 class predictions with confidence scores
    - **Grad-CAM heatmap** showing which parts of the image the model focused on
    - Model accuracy from training
    """)

    if st.session_state.cv_dataset_id and st.session_state.cv_training_results:
        dataset_id = st.session_state.cv_dataset_id

        st.info(f"📦 **Using model trained on dataset:** `{dataset_id}`")

        # Upload test image
        uploaded_test_image = st.file_uploader(
            "Choose an image to classify",
            type=["jpg", "jpeg", "png", "bmp"],
            help="Upload an image from the same domain as your training data"
        )

        if uploaded_test_image is not None:
            # Display uploaded image
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📷 Input Image")
                st.image(uploaded_test_image, use_column_width=True)

            # Make prediction
            with col2:
                st.subheader("🔮 Predictions")

                if st.button("Predict", type="primary"):
                    with st.spinner("Analyzing image..."):
                        try:
                            # Send prediction request
                            files = {
                                "file": (uploaded_test_image.name, uploaded_test_image.getvalue())
                            }
                            response = requests.post(
                                f"{API_BASE_URL}/cv/predict/{dataset_id}",
                                files=files
                            )

                            if response.status_code == 200:
                                result = response.json()
                                predictions = result["predictions"]
                                model_info = result["model_info"]

                                # Show model info
                                st.success(f"✅ Model: **{model_info['model_name']}** (Accuracy: {model_info['accuracy']:.2%})")

                                # Show predictions
                                st.markdown("### Top Predictions")
                                for i, pred in enumerate(predictions, 1):
                                    confidence = pred['confidence']
                                    color = "🟢" if i == 1 else "🟡" if i == 2 else "🟠"
                                    st.markdown(
                                        f"{color} **{i}. {pred['class']}** - {confidence:.2%}"
                                    )
                                    st.progress(confidence)

                            else:
                                st.error(f"Prediction failed: {response.text}")

                        except Exception as e:
                            st.error(f"Error making prediction: {str(e)}")

            # Show confusion matrix if available
            if st.session_state.cv_training_results:
                summary = st.session_state.cv_training_results.get("summary", {})
                if "confusion_matrix" in summary:
                    st.markdown("---")
                    st.subheader("📊 Confusion Matrix & Per-Class Metrics")

                    cm_data = summary["confusion_matrix"]
                    class_names = cm_data["class_names"]
                    cm = cm_data["confusion_matrix"]

                    # Display confusion matrix
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        import plotly.figure_factory as ff
                        import numpy as np

                        fig = ff.create_annotated_heatmap(
                            z=cm,
                            x=class_names,
                            y=class_names,
                            colorscale='Blues',
                            showscale=True
                        )

                        fig.update_layout(
                            title=f"Confusion Matrix - {cm_data['model_name']}",
                            xaxis_title="Predicted",
                            yaxis_title="Actual",
                            height=400
                        )

                        st.plotly_chart(fig, use_column_width=True)

                    with col2:
                        st.markdown("#### Per-Class Metrics")
                        metrics_df = pd.DataFrame(cm_data["per_class_metrics"])
                        st.dataframe(
                            metrics_df.style.format({
                                "precision": "{:.3f}",
                                "recall": "{:.3f}",
                                "f1_score": "{:.3f}"
                            }),
                            width='content',
                            hide_index=True
                        )

    else:
        st.warning("⚠️ Please upload a dataset and train models first!")
        st.markdown("""
        **Next steps:**
        1. Go to **Upload Dataset** tab
        2. Upload your image dataset (ZIP file)
        3. Go to **Train Models** tab
        4. Configure and start training
        5. Come back here to make predictions!
        """)
