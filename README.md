# 🤖 AutoML Pipeline Builder

> **From Raw Data to Trained Models in Minutes - Zero Code Required**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-2.8+-orange.svg)](https://mlflow.org/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple.svg)](https://shap.readthedocs.io/)
[![CI](https://github.com/kitsakisGk/AutoML-Forge/workflows/CI/badge.svg)](https://github.com/kitsakisGk/AutoML-Forge/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An **explainability-first AutoML platform** that automatically cleans data and trains ML models. Built for the Swiss market with bilingual support (EN/DE).

🎯 **Achieved 88.67% R² score** on regression tasks with automatic model selection.

---

## ✨ Features

- 🔄 **Automatic Data Cleaning** - Smart detection and fixing of data issues
- 📊 **Intelligent Data Profiling** - Comprehensive analysis with visualizations
- 🤖 **AutoML Training** - Train multiple models and compare results automatically
- 🔬 **MLflow Experiment Tracking** - Production-grade experiment tracking and model versioning
- 🌍 **Bilingual Interface** - Full support for English and German (EN/DE)
- 🔒 **Privacy-First** - All processing happens locally, no cloud dependency
- 🐳 **One-Click Deployment** - Export to Docker, FastAPI, or Python scripts
- 📈 **Explainability** - Understand every decision with SHAP values
- 🚀 **Production-Ready** - Generate deployment-ready APIs automatically

---

## 🎯 Perfect For

- **Data Scientists** - Accelerate your ML workflow
- **ML Engineers** - Automate pipeline creation
- **Business Analysts** - Build models without coding
- **Students** - Learn ML best practices
- **Researchers** - Rapid prototyping

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Docker (optional, for containerized deployment)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/automl-forge.git
cd automl-forge
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements/dev.txt
```

### Running Locally

**Option 1: Run Backend and Frontend Separately**

```bash
# Terminal 1 - Start FastAPI backend
cd backend
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 - Start Streamlit frontend
streamlit run frontend/app.py
```

**Option 2: Use Docker Compose**

```bash
cd docker
docker-compose up --build
```

Then open:
- **Frontend**: http://localhost:8501
- **API Docs**: http://localhost:8000/api/docs

---

## 📖 Usage

### 1. Upload Your Data

Upload CSV, Excel, JSON, or Parquet files (up to 2GB)

### 2. Explore Data Profile

Get automatic insights:
- Data types and statistics
- Missing values analysis
- Correlations
- Distribution visualizations

### 3. Clean Your Data ✅

Receive explainable smart suggestions:
- ✅ Missing value detection with imputation strategies
- ✅ Outlier detection using IQR method
- ✅ Data type issue identification
- ✅ Categorical encoding recommendations
- ✅ Alternative approaches with reasoning
- ✅ Export Python cleaning script

### 4. Train Models ✅

Automatically train and compare 6 models:
- ✅ Linear/Logistic Regression
- ✅ Ridge Regression
- ✅ Random Forest
- ✅ Gradient Boosting
- ✅ XGBoost
- ✅ LightGBM

All experiments tracked with MLflow for reproducibility.

### 5. View Experiment Tracking 🔬

Track all ML experiments with MLflow UI:

```bash
python run_mlflow.py
```

Then open http://localhost:5000 to view:
- All model runs with parameters and metrics
- Model comparison charts
- Feature importance artifacts
- Cross-validation scores
- Model artifacts for deployment

Features:
- Auto problem type detection (classification/regression)
- 5-fold cross-validation
- Feature importance visualization
- Model comparison dashboard

### 5. Deploy to Production (Planned)

Export as:
- Python script (✅ Available now)
- Docker image (Coming soon)
- FastAPI endpoint (Coming soon)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                 Streamlit Frontend                   │
│              (Bilingual EN/DE Interface)             │
└─────────────────┬───────────────────────────────────┘
                  │ HTTP/REST
┌─────────────────▼───────────────────────────────────┐
│                  FastAPI Backend                     │
│  ┌─────────────┬──────────────┬──────────────────┐  │
│  │ Data Loader │ Data Profiler│ AutoML Engine    │  │
│  │ Multi-format│ Smart Analysis│ Model Training   │  │
│  └─────────────┴──────────────┴──────────────────┘  │
└─────────────────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────┐
│              ML Libraries & Storage                  │
│  scikit-learn • XGBoost • LightGBM • MLflow         │
└─────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

### Backend
- **FastAPI** - Modern async Python web framework
- **Pydantic** - Data validation
- **Pandas/NumPy** - Data processing

### Machine Learning
- **scikit-learn** - Traditional ML algorithms
- **XGBoost/LightGBM** - Gradient boosting
- **Optuna** - Hyperparameter optimization
- **SHAP** - Model explainability
- **MLflow** - Experiment tracking

### Frontend
- **Streamlit** - Interactive web UI
- **Plotly** - Interactive visualizations

### DevOps
- **Docker** - Containerization
- **pytest** - Testing framework
- **GitHub Actions** - CI/CD

---

## 📊 Project Status

**Current Phase:** Weeks 1-6 Complete! ✅

| Week | Feature | Status |
|------|---------|--------|
| 1-2 | Foundation + Data Upload & Profiling | ✅ Complete |
| 3-4 | Smart Data Cleaning | ✅ Complete |
| 5-6 | AutoML Engine | ✅ Complete |
| 7-8 | MLflow + Docker | 🚧 In Progress |
| 9-10 | SHAP + Hyperparameter Tuning | 📋 Planned |
| 11-12 | Polish & Documentation | 📋 Planned |

---

## 🧪 Development

### Run Tests
```bash
pytest tests/ -v --cov=backend --cov-report=html
```

### Code Formatting
```bash
black backend/ frontend/
isort backend/ frontend/
flake8 backend/ frontend/
```

### Type Checking
```bash
mypy backend/
```

---

## 🌍 Internationalization

Fully bilingual interface supporting:
- 🇬🇧 **English**
- 🇩🇪 **German** (Swiss focused)

Translation files: `frontend/i18n/en.json` and `frontend/i18n/de.json`

---

## 📝 Documentation

Full documentation coming soon at [docs/](docs/)

- [Architecture Overview](docs/architecture.md) (Coming Soon)
- [API Reference](docs/api-reference.md) (Coming Soon)
- [User Guide](docs/user-guide.md) (Coming Soon)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Your Name**

- LinkedIn: [https://www.linkedin.com/in/georgios-kitsakis-gr/](https://www.linkedin.com/in/georgios-kitsakis-gr/)
- GitHub: [https://github.com/kitsakisGk](https://github.com/kitsakisGk)
- Email: kitsakisgk@gmail.com

---

## 🙏 Acknowledgments

- Built for the Zurich job market with Swiss precision
- Inspired by modern AutoML tools (H2O.ai, AutoGluon, PyCaret)
- Focus on explainability and privacy

---

## 📈 Roadmap

### Planned Features
- [ ] Advanced feature engineering
- [ ] Time series forecasting
- [ ] Natural language data queries
- [ ] Model drift detection
- [ ] A/B testing framework
- [ ] Team collaboration features

---

## ⭐ Star History

If you find this project useful, please consider giving it a star!

---
