# 🚀 Quick Start Guide

Get AutoML Forge running in 5 minutes!

---

## Prerequisites

- **Python 3.10+** installed (3.11 or 3.12 recommended)
- **Git** installed
- **8GB RAM minimum**

---

## Option 1: Automated Setup (Windows)

### Step 1: Run Setup Script

Double-click `setup.bat` or run in terminal:

```bash
setup.bat
```

This will:
- Create a virtual environment
- Install all dependencies
- Set up the project

### Step 2: Run the Application

**Terminal 1 - Backend API:**
```bash
venv\Scripts\activate
py run_backend.py
```

**Terminal 2 - Frontend UI:**
```bash
venv\Scripts\activate
streamlit run frontend/app.py
```

### Step 3: Open in Browser

- **Frontend**: http://localhost:8501
- **API Docs**: http://localhost:8000/api/docs

---

## Option 2: Manual Setup

### Step 1: Clone Repository

```bash
git clone https://github.com/kitsakisGk/AutoML-Forge.git
cd AutoML-Forge
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
py -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements/base.txt
```

### Step 4: Run Backend

```bash
py run_backend.py
```

Or manually:
```bash
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Step 5: Run Frontend (New Terminal)

```bash
venv\Scripts\activate  # Activate virtual environment again
streamlit run frontend/app.py
```

---

## Option 3: Docker

### Build and Run with Docker Compose

```bash
cd docker
docker-compose up --build
```

**Access:**
- Frontend: http://localhost:8501
- Backend API: http://localhost:8000/api/docs

---

## Testing the Application

### 1. Upload Sample Data

We've included a sample dataset at `tests/fixtures/sample_data.csv`

1. Open http://localhost:8501
2. Go to "📁 Upload Data" tab
3. Upload `tests/fixtures/sample_data.csv`
4. Click "Upload & Analyze"

### 2. View Data Profile

1. After upload, go to "📊 Data Profile" tab
2. Click "Generate Profile"
3. See comprehensive data analysis

### 3. Switch Language

Use the language selector in the sidebar to switch between English and German!

---

## Running Tests

```bash
# Activate virtual environment first
venv\Scripts\activate

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=backend --cov-report=html

# View coverage report
start htmlcov/index.html  # Windows
open htmlcov/index.html   # Mac
xdg-open htmlcov/index.html  # Linux
```

---

## Troubleshooting

### Issue: "Module not found" error

**Solution:**
```bash
# Make sure you're in the project root directory
cd AutoML-Forge

# Activate virtual environment
venv\Scripts\activate

# Reinstall dependencies
pip install -r requirements/base.txt
```

### Issue: "Port already in use"

**Solution for Backend (port 8000):**
```bash
# Find process using port 8000
netstat -ano | findstr :8000

# Kill the process (replace PID with actual process ID)
taskkill /PID <PID> /F
```

**Solution for Frontend (port 8501):**
```bash
# Find process using port 8501
netstat -ano | findstr :8501

# Kill the process
taskkill /PID <PID> /F
```

### Issue: Frontend can't connect to backend

**Check:**
1. Backend is running on http://localhost:8000
2. Visit http://localhost:8000/api/health
3. Should see: `{"status":"healthy",...}`

If not working, restart backend and check for errors.

### Issue: "Permission denied" on Windows

**Solution:**
- Use `py` instead of `python` or `python3`
- Make sure Python is installed from python.org, not from Microsoft Store

---

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/api/health
```

### Upload File
```bash
curl -X POST "http://localhost:8000/api/upload" \
  -F "file=@tests/fixtures/sample_data.csv"
```

### Get Profile
```bash
curl http://localhost:8000/api/profile/sample_data.csv
```

### Interactive API Docs
Visit: http://localhost:8000/api/docs

---

## Next Steps

1. ✅ **Upload your own data** - Try CSV, Excel, JSON, or Parquet files
2. ✅ **Explore data profiling** - See statistics, missing values, correlations
3. 🔜 **Data cleaning** (Coming in Week 3-4)
4. 🔜 **AutoML training** (Coming in Week 5-6)
5. 🔜 **Export to production** (Coming in Week 7)

---

## Project Structure

```
AutoML-Forge/
├── backend/          # FastAPI backend
│   ├── api/         # API routes
│   └── core/        # Business logic
├── frontend/        # Streamlit UI
│   └── i18n/        # Translations (EN/DE)
├── tests/           # Unit tests
│   └── fixtures/    # Sample data
├── docker/          # Docker configuration
└── requirements/    # Python dependencies
```

---

## Useful Commands

```bash
# Activate virtual environment
venv\Scripts\activate

# Run backend
py run_backend.py

# Run frontend
streamlit run frontend/app.py

# Run tests
pytest tests/ -v

# Code formatting
black backend/ frontend/

# Type checking
mypy backend/

# Deactivate virtual environment
deactivate
```

---

## Getting Help

- 📖 **Documentation**: See [README.md](README.md)
- 🐛 **Issues**: https://github.com/kitsakisGk/AutoML-Forge/issues
- 📧 **Contact**: Create an issue on GitHub

---

## What's Working Now?

✅ **Week 1-2 Features:**
- Multi-format data upload (CSV, Excel, JSON, Parquet)
- Comprehensive data profiling
- Bilingual UI (English/German)
- REST API with documentation
- Docker deployment

🔜 **Coming Soon:**
- Smart data cleaning (Week 3-4)
- AutoML training (Week 5-6)
- Production export (Week 7)

---

**Ready to build ML pipelines? Start uploading data!** 🚀
