@echo off
echo ========================================
echo AutoML Forge - Setup Script
echo ========================================
echo.

echo [1/4] Creating virtual environment...
py -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)
echo Virtual environment created successfully!
echo.

echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat
echo.

echo [3/4] Upgrading pip...
python -m pip install --upgrade pip
echo.

echo [4/4] Installing dependencies...
pip install -r requirements\base.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo.

echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo Next steps:
echo 1. Activate the virtual environment: venv\Scripts\activate
echo 2. Run backend: py run_backend.py
echo 3. Run frontend: streamlit run frontend/app.py
echo.
pause
