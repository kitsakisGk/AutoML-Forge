# 🤖 AutoML Pipeline Builder

> **From Raw Data to Trained Models in Minutes - Zero Code Required**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-2.8+-orange.svg)](https://mlflow.org/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple.svg)](https://shap.readthedocs.io/)
[![CI](https://github.com/kitsakisGk/AutoML-Forge/workflows/CI/badge.svg)](https://github.com/kitsakisGk/AutoML-Forge/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An **explainability-first AutoML platform** that automatically cleans data and trains ML models. Built with FastAPI and Streamlit, featuring MLflow experiment tracking and SHAP model interpretability.

---

## 🎯 What is This?

AutoML Pipeline Builder is a complete machine learning automation platform that takes you from raw data to production-ready models in minutes. Upload your data, get smart cleaning suggestions with explanations, and automatically train 6 different ML models with full experiment tracking.

**Key Highlights:**
- ✅ Automatic problem detection (classification vs regression)
- ✅ Smart data cleaning with explainability - understand every suggestion
- ✅ Train 6 models automatically: Linear/Ridge, Random Forest, Gradient Boosting, XGBoost, LightGBM
- ✅ MLflow experiment tracking for reproducibility
- ✅ SHAP values for model interpretability
- ✅ Bilingual interface (EN/DE) for Swiss market
- ✅ Production-ready REST API with OpenAPI documentation

---

## ✨ Features

### 📤 Data Upload
Upload CSV, Excel, JSON, or Parquet files with automatic format detection.

![Upload Data](tests/Screenshots/Upload_data.jpg)

### 📊 Data Profiling
Comprehensive automatic analysis with statistics, correlations, distributions, and missing value detection.

![Data Profile](tests/Screenshots/Data_Profile.jpg)

### 🧹 Smart Data Cleaning
Explainability-first cleaning suggestions. Every recommendation comes with reasoning, alternatives, and impact analysis.

![Data Cleaning - Suggestions](tests/Screenshots/Data_Cleaning_1.jpg)
![Data Cleaning - Details](tests/Screenshots/Data_Cleaning_2.jpg)
![Data Cleaning - After](tests/Screenshots/Data_Cleaning_3_after_cleaning.jpg)

### 🤖 AutoML Training
Automatic model selection and training with 5-fold cross-validation. Compare 6 different models side-by-side.

![Train Models - Selection](tests/Screenshots/Train_models_1.jpg)
![Train Models - Results](tests/Screenshots/Train_models_2.jpg)
![Train Models - Comparison](tests/Screenshots/Train_models_3.jpg)

### 🔬 MLflow Integration
Track every experiment with parameters, metrics, and model artifacts for full reproducibility.

### 🎓 SHAP Explainability
Understand why your models make predictions with SHAP (SHapley Additive exPlanations) values.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/kitsakisGk/AutoML-Forge.git
cd AutoML-Forge
```

2. **Create and activate virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements/base.txt
```

### Running the Application

**Option 1: Run both backend and frontend separately**

Terminal 1 - Backend:
```bash
python run_backend.py
```

Terminal 2 - Frontend:
```bash
streamlit run frontend/app.py
```

**Option 2: Run everything together**
```bash
python start_all.py
```

Then open:
- **Application UI**: http://localhost:8501
- **API Documentation**: http://localhost:8000/api/docs

---

## 📖 How to Use

### 1. Upload Your Data
Upload CSV, Excel, JSON, or Parquet files (up to 2GB). The platform automatically detects file format and encoding.

### 2. Explore Data Profile
Get automatic insights including:
- Summary statistics (mean, median, std, min, max)
- Missing values analysis
- Feature correlations
- Distribution visualizations
- Data type detection

### 3. Clean Your Data
Receive smart cleaning suggestions with full explanations:
- **Missing values**: Imputation strategies with reasoning (median for outliers, mean for normal distributions)
- **Outliers**: IQR-based detection with impact analysis
- **Data types**: Automatic type correction recommendations
- **Alternatives**: Multiple approaches for every issue

Each suggestion explains:
- ✅ What the issue is
- ✅ Why the fix is recommended
- ✅ What alternatives exist
- ✅ What the impact will be

### 4. Train Models
Automatically train and compare 6 different models:

**Classification:**
- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

**Regression:**
- Linear Regression
- Ridge Regression
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

**Features:**
- Automatic problem type detection
- 5-fold cross-validation for robust evaluation
- Feature importance analysis
- Model comparison dashboard
- Best model selection based on primary metric (accuracy/R²)

### 5. View Experiment Tracking

Track all ML experiments with MLflow:

```bash
# Terminal viewer (always works)
python view_mlflow_results.py

# MLflow web UI (may show blank page on Windows)
python run_mlflow.py
```

View:
- All model runs with parameters and metrics
- Cross-validation scores
- Feature importance artifacts
- Model artifacts for deployment

---

## 🛠️ Tech Stack

**Backend:**
- FastAPI - Modern async Python web framework
- Pydantic - Data validation and settings
- scikit-learn - Traditional ML algorithms
- XGBoost/LightGBM - Gradient boosting frameworks
- SHAP - Model explainability
- MLflow - Experiment tracking and model versioning

**Frontend:**
- Streamlit - Interactive web UI
- Plotly - Interactive visualizations
- i18n support - English/German translations

**Data Processing:**
- Pandas/NumPy - Data manipulation
- ydata-profiling - Automated profiling

**DevOps:**
- Docker - Containerization
- pytest - Testing framework
- GitHub Actions - CI/CD pipeline

---

## 🌍 Bilingual Support

Fully bilingual interface supporting:
- 🇬🇧 **English**
- 🇩🇪 **German** (Swiss market focus)

Translation files: `frontend/i18n/en.json` and `frontend/i18n/de.json`

---

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=backend --cov-report=html
```

Tests are automatically run on GitHub Actions for Python 3.9, 3.10, and 3.11.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Kitsakis Giorgos**

- LinkedIn: [https://www.linkedin.com/in/georgios-kitsakis-gr/](https://www.linkedin.com/in/georgios-kitsakis-gr/)
- GitHub: [https://github.com/kitsakisGk](https://github.com/kitsakisGk)
- Email: kitsakisgk@gmail.com

---

## 🙏 Acknowledgments

- Inspired by modern AutoML tools (H2O.ai, AutoGluon, PyCaret)
- Focus on explainability and production-readiness
- Built with attention to code quality and best practices
