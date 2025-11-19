# 🚀 Quick Start Guide

## Option 1: Start Everything Together (Recommended)

### Run the complete app:
```bash
python start_all.py
```

This will start:
- ✅ FastAPI Backend (http://localhost:8000)
- ✅ Streamlit Frontend (http://localhost:8501)
- 🌐 Opens browser automatically

---

## Option 2: Run Services Separately

### Terminal 1 - Backend API:
```bash
python run_backend.py
```

### Terminal 2 - Streamlit UI:
```bash
streamlit run frontend/app.py
```

### Terminal 3 - MLflow UI (Optional):
```bash
python run_mlflow.py
```

---

## 📊 View MLflow Experiments

### Method 1: Web UI (if it works)
```bash
python run_mlflow.py
```
Then open: http://127.0.0.1:5000

### Method 2: Terminal Viewer (Always works!)
```bash
python view_mlflow_results.py
```

This shows all your experiments in a nice table format directly in the terminal!

---

## 🎯 Using the App

1. **Upload Data** - CSV, Excel, JSON, or Parquet
2. **Profile Data** - See statistics and visualizations
3. **Clean Data** - Get smart suggestions and apply fixes
4. **Train Models** - Automatically train and compare 6 models
5. **View Results** - See best model and all metrics

All experiments are automatically tracked with MLflow!

---

## 🔍 Troubleshooting

### MLflow UI shows blank page?
No problem! Use the terminal viewer:
```bash
python view_mlflow_results.py
```

### Port already in use?
Kill Python processes:
```bash
taskkill /F /IM python.exe
```
Then restart the services.

### Need to reset everything?
```bash
# Delete uploaded files
rm -rf tmp/

# Delete MLflow experiments (careful!)
rm -rf mlruns/
```

---

## 📁 Project Structure

```
AutoML_Data_Pipeline_Builder/
├── backend/
│   ├── api/          # FastAPI routes
│   └── core/         # ML engines
├── frontend/
│   └── app.py        # Streamlit UI
├── start_all.py      # 🚀 Start everything
├── run_mlflow.py     # Launch MLflow UI
└── view_mlflow_results.py  # View experiments in terminal
```

---

## ⚡ Quick Commands

```bash
# Start everything
python start_all.py

# View MLflow results
python view_mlflow_results.py

# Test MLflow integration
python test_mlflow_integration.py

# Run tests
pytest tests/ -v

# Check git status
git status
```

---

**Built with ❤️ for the Zurich job market**
