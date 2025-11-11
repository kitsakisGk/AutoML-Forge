# 🚀 Quick Start - Testing Your AutoML Forge

## Step 1: Start Backend (Terminal 1)

```bash
# Activate virtual environment
venv\Scripts\activate

# Start FastAPI backend
py run_backend.py
```

**Expected output:**
```
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Test it:** Open http://localhost:8000/api/docs in your browser

---

## Step 2: Start Frontend (Terminal 2)

Open a NEW terminal window:

```bash
# Activate virtual environment
venv\Scripts\activate

# Start Streamlit frontend
streamlit run frontend/app.py
```

**Expected output:**
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

**App will open automatically in your browser!**

---

## Step 3: Test the Features! 🎉

### 1. Upload Data
- Go to "📁 Upload Data" tab
- Upload `tests/fixtures/sample_data.csv`
- Click "Upload & Analyze"
- See file info (15 rows, 5 columns)

### 2. View Data Profile
- Go to "📊 Data Profile" tab
- Click "Generate Profile"
- See:
  - Basic stats
  - Missing values (you'll see 2 columns with missing data!)
  - Data types
  - Memory usage

### 3. **SMART DATA CLEANING** (The WOW Feature!)
- Go to "🧹 Clean Data" tab
- Click "Analyze Data Issues"
- Watch the magic! You'll see suggestions like:

```
🟡 Missing values in 'income'
13% of values are missing (2 rows)

💡 Suggestion: Impute with median (75,000)
📝 Reason: Numerical column with outliers. Median is more robust...

Alternatives:
• Drop rows (-2 rows)
• Use mean (72,500)
• Use KNN imputation

[✓] Apply this fix
```

- Check the boxes for fixes you want
- Click "Apply Selected Fixes"
- **Download the Python script!** 📥

### 4. Switch Language!
- Use the sidebar dropdown
- Switch between English 🇬🇧 and German 🇩🇪

---

## Troubleshooting

### Backend won't start?
```bash
# Check if port 8000 is in use
netstat -ano | findstr :8000

# If something is using it, kill it:
taskkill /PID <PID> /F
```

### Frontend can't connect?
- Make sure backend is running (http://localhost:8000/api/health should work)
- Check API_BASE_URL in `frontend/app.py` is `http://localhost:8000/api`

### Import errors?
```bash
# Make sure you're in virtual environment
venv\Scripts\activate

# Reinstall dependencies
pip install -r requirements/base.txt
```

---

## What to Look For 🔍

**In the Clean Data tab, you should see:**

1. **Missing value suggestion** for 'income' column
2. **Missing value suggestion** for 'education' column
3. Possibly **encoding suggestions** for categorical columns
4. Beautiful color-coded severity indicators
5. Detailed explanations for each suggestion
6. Alternative approaches

**Try this:**
- Select a few suggestions
- Click "Apply Selected Fixes"
- Go back to "Data Profile" to see the cleaned data!
- Download the Python script and check it out!

---

## Next Steps

Once you've tested everything:
1. Try uploading your own CSV file
2. See what issues it finds
3. Apply some cleaning
4. Download the generated Python script

**This is ready to show on LinkedIn!** 🚀

---

**Need help?** Check QUICKSTART.md for more details.
