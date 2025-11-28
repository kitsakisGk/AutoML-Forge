@echo off
echo ========================================
echo AutoML Forge - HuggingFace Deployment
echo ========================================
echo.

REM Check if HF space is already cloned
if not exist "hf_space" (
    echo Cloning HuggingFace Space...
    git clone https://huggingface.co/spaces/kitsakisG/automl-forge hf_space
)

cd hf_space

echo.
echo Copying files...
copy ..\app_hf.py app.py
copy ..\requirements_hf.txt requirements.txt

echo.
echo Creating README...
(
echo ---
echo title: AutoML Forge CV Demo
echo emoji: 📸
echo colorFrom: blue
echo colorTo: green
echo sdk: streamlit
echo sdk_version: 1.28.0
echo app_file: app.py
echo pinned: false
echo license: mit
echo ---
echo.
echo # 📸 AutoML Forge - Computer Vision Demo
echo.
echo State-of-the-art image classification with **Vision Transformers** and **CNNs**.
echo.
echo ## 🎯 Try It Out
echo.
echo 1. **Select a model** from the dropdown
echo 2. **Upload an image** ^(JPG, PNG^)
echo 3. **Click Predict** to get top-5 predictions
echo.
echo ## 🤖 Models Available
echo.
echo - **MobileNetV3**: Fast, mobile-optimized ^(5.4M params^)
echo - **ResNet18**: Lightweight CNN ^(11.7M params^)
echo - **EfficientNet-B0**: Balanced ^(5.3M params^)
echo - **ViT-Base**: State-of-the-art Transformer ^(86M params^)
echo.
echo All models pre-trained on **ImageNet** ^(1.2M images, 1000 classes^).
echo.
echo ## 🚀 Full Platform
echo.
echo See the full platform at: [GitHub](https://github.com/kitsakisGk/AutoML-Forge^)
echo.
echo ## 👤 Author
echo.
echo **Kitsakis Giorgos**
echo - GitHub: [@kitsakisGk](https://github.com/kitsakisGk^)
echo - LinkedIn: [georgios-kitsakis-gr](https://www.linkedin.com/in/georgios-kitsakis-gr/^)
) > README.md

REM Remove old template files if they exist
if exist "src" rmdir /s /q src
if exist "Dockerfile" del Dockerfile

echo.
echo Adding files to git...
git add .

echo.
echo Current status:
git status

echo.
echo ========================================
echo Files are ready! Now run:
echo.
echo   cd hf_space
echo   git commit -m "Deploy AutoML Forge CV Demo"
echo   git push
echo.
echo You will be prompted for:
echo   Username: kitsakisG
echo   Password: [paste your HuggingFace token]
echo.
echo Get token from: https://huggingface.co/settings/tokens
echo ========================================

pause
