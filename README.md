# 🤖 AutoML Pipeline Builder

> **From Raw Data to Production Model in Minutes - No Code Required**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**AutoML Pipeline Builder** is an open-source platform that automatically creates ETL pipelines and trains machine learning models with zero coding required. Built with explainability and privacy in mind.

---

## ✨ Features

- 🔄 **Automatic Data Cleaning** - Smart detection and fixing of data issues
- 📊 **Intelligent Data Profiling** - Comprehensive analysis with visualizations
- 🤖 **AutoML Training** - Train multiple models and compare results automatically
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

### 3. Clean Your Data (Coming in Week 3-4)

Receive smart suggestions for:
- Handling missing values
- Outlier detection
- Feature encoding
- Data type corrections

### 4. Train Models (Coming in Week 5-6)

Automatically train and compare:
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- Neural Networks

### 5. Deploy to Production (Coming in Week 7)

Export as:
- Docker image
- FastAPI endpoint
- Python script
- ONNX model

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

**Current Phase:** Week 1-2 Foundation ✅

| Week | Feature | Status |
|------|---------|--------|
| 1-2 | Foundation + Data Upload & Profiling | ✅ In Progress |
| 3-4 | Smart Data Cleaning | 🔜 Coming Soon |
| 5-6 | AutoML Engine | 🔜 Coming Soon |
| 7 | Production Export | 🔜 Coming Soon |
| 8 | Polish & Launch | 🔜 Coming Soon |

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

- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- GitHub: [@YourGitHub](https://github.com/yourusername)
- Email: your.email@example.com

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

**Built with ❤️ for the data science community**
