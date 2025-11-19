# 📁 Project Structure

## 🎯 Essential Files (What You Need to Know)

### 🚀 **Startup Scripts**
- **`start_all.py`** - ⭐ Main script! Starts backend + frontend together
- **`run_backend.py`** - Start only FastAPI backend
- **`run_mlflow.py`** - Launch MLflow UI (for viewing experiments)
- **`view_mlflow_results.py`** - View MLflow results in terminal (always works!)

### 📚 **Documentation**
- **`README.md`** - Main project documentation
- **`QUICK_START.md`** - Quick commands to get started
- **`INTERVIEW_GUIDE.md`** - Technical interview preparation
- **`PROGRESS.md`** - Development progress tracking

### 🧪 **Testing**
- **`test_mlflow_integration.py`** - Test MLflow tracking
- **`tests/`** - All unit tests

### ⚙️ **Configuration**
- **`.env.example`** - Environment variables template
- **`pyproject.toml`** - Python project configuration
- **`requirements/`** - Python dependencies
  - `base.txt` - Core dependencies
  - `dev.txt` - Development dependencies

---

## 📂 Main Directories

### **`backend/`** - FastAPI Backend
```
backend/
├── api/
│   ├── main.py          # FastAPI app
│   └── routes/          # API endpoints
│       ├── upload.py    # File upload
│       ├── profile.py   # Data profiling
│       ├── clean.py     # Data cleaning
│       └── train.py     # Model training
└── core/
    ├── data_loader.py   # Load CSV/Excel/JSON/Parquet
    ├── data_profiler.py # Generate data statistics
    ├── data_cleaner.py  # Smart cleaning suggestions
    └── automl_engine.py # ML model training
```

### **`frontend/`** - Streamlit UI
```
frontend/
├── app.py               # Main Streamlit app
└── i18n/                # Translations
    ├── en.json          # English
    └── de.json          # German
```

### **`docker/`** - Docker Configuration
- Docker Compose setup
- Dockerfile for containerization

### **`tests/`** - Unit Tests
- `test_core/` - Core functionality tests
- `test_api/` - API endpoint tests
- `fixtures/` - Test data

### **`.github/`** - CI/CD
- `workflows/` - GitHub Actions for automated testing

---

## 🗂️ Generated/Ignored Directories

### **`mlruns/`** (ignored)
- MLflow experiment tracking data
- View with: `python view_mlflow_results.py`

### **`tmp/`** (ignored)
- Uploaded files storage
- Temporary data processing

### **`venv/`** (ignored)
- Python virtual environment
- All dependencies installed here

---

## 🎨 Simple Workflow

### 1. **Start the App**
```bash
python start_all.py
```

### 2. **Use the App**
- Upload → Profile → Clean → Train
- Everything tracked automatically!

### 3. **View Results**
```bash
python view_mlflow_results.py
```

---

## 🧹 What Was Cleaned Up

Removed old/duplicate directories:
- ❌ `backendapi/`, `backendcore/`, `backendmodels/`, `backendutils/`
- ❌ `frontendcomponents/`, `frontendi18n/`, `frontendpages/`
- ❌ `testsfixtures/`, `teststest_api/`, `teststest_core/`

Removed duplicate files:
- ❌ `QUICKSTART.md` (use `QUICK_START.md` instead)
- ❌ `Roadmap_Refined.md`
- ❌ `START_HERE.md`
- ❌ `test_api.py` (use `tests/` directory)

---

## 📊 Key Components

### **AutoML Engine** ([backend/core/automl_engine.py](backend/core/automl_engine.py))
- Automatic model selection
- Trains 6 models automatically
- MLflow integration
- Feature importance
- Cross-validation

### **Data Cleaner** ([backend/core/data_cleaner.py](backend/core/data_cleaner.py))
- Smart cleaning suggestions
- Explainability first
- Handles missing values, outliers, types

### **Streamlit UI** ([frontend/app.py](frontend/app.py))
- 4 tabs: Upload, Profile, Clean, Train
- Bilingual (EN/DE)
- Interactive visualizations

---

## 🔧 Maintenance

### Clean temporary files:
```bash
rm -rf tmp/
rm -rf mlruns/
```

### Reinstall dependencies:
```bash
pip install -r requirements/dev.txt
```

### Run tests:
```bash
pytest tests/ -v
```

---

**Last Updated:** November 19, 2025 (Week 7 - MLflow Integration Complete!)
